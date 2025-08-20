from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional, E, Delta_t, F):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,) or (L, P)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
            E          (complex64): learnable matrix for Lambda extension (P, R)
            Delta_t    (complex64): time-varying diagonal matrices        (L, R, R)
            F          (complex64): learnable matrix for Lambda extension (R, P)
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    L = input_sequence.shape[0]
    
    # Lambda extension: Lambda_ex = Lambda_bar + E @ Delta_t @ F

    # E: (P, R), Delta_t: (L, R, R), F: (R, P)
    # For each time step t, compute E @ Delta_t[t] @ F
    def compute_lambda_extension(delta_t):
        # delta_t: (R, R), E: (P, R), F: (R, P)
        return E @ delta_t @ F  # (P, P)
    
    # Apply to all time steps: (L, P, P)
    Lambda_extensions = jax.vmap(compute_lambda_extension)(Delta_t)
    
    # Add to base Lambda_bar
    if len(Lambda_bar.shape) == 1:
        # Lambda_bar is (P,), broadcast to (L, P)
        base_Lambda = Lambda_bar * np.ones((L, Lambda_bar.shape[0]))
    else:
        base_Lambda = Lambda_bar
        
    # Extract diagonal elements from Lambda_extensions and add to base_Lambda
    Lambda_elements = base_Lambda + jax.vmap(np.diag)(Lambda_extensions)


    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (Lambda_elements, Bu_elements),
                                          reverse=True)
        xs = np.concatenate((xs, xs2), axis=-1)
    
    if C_tilde is None:
        return xs, None

    if conj_sym:
        return xs, jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return xs, jax.vmap(lambda x: (C_tilde @ x).real)(xs)
        


class ExtendedS5SSM(nn.Module):
    Lambda_re_init: np.DeviceArray
    Lambda_im_init: np.DeviceArray
    V: np.DeviceArray
    Vinv: np.DeviceArray

    H: int
    P: int
    R: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = True
    bidirectional: bool = True
    step_rescale: float = 1.0

    """ The S5 SSM with Lambda Extension
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            R           (int32):     Rank for Delta_t
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize E matrix (P, R)
        E_init = normal(stddev=0.01)
        self.E_real = self.param("E_real", E_init, (self.P, self.R))
        self.E_imag = self.param("E_imag", E_init, (self.P, self.R))
        self.E = self.E_real + 1j * self.E_imag
        
        # Initialize F matrix (R, P)  
        F_init = normal(stddev=0.01)
        self.F_real = self.param("F_real", F_init, (self.R, self.P))
        self.F_imag = self.param("F_imag", F_init, (self.R, self.P))
        self.F = self.F_real + 1j * self.F_imag


        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          self.Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param("C1",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)
                self.C2 = self.param("C2",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param("C",
                                    lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                    C_shape)

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])

        # Initialize Delta_t_diag if using Lambda extension
        if self.R is not None:
            # Initialize with a reasonable maximum sequence length
            # This will be dynamically adjusted in __call__
            max_seq_len = 1024  # You can adjust this based on your dataset
            self.Delta_t_diag = self.param("Delta_t_diag", 
                                          normal(stddev=0.01), 
                                          (max_seq_len, self.R))

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        L = input_sequence.shape[0]
        
        # Get Delta_t if using Lambda extension
        if self.R is not None:
            # Use the pre-initialized Delta_t_diag and slice it to the current sequence length
            if L <= self.Delta_t_diag.shape[0]:
                Delta_t_diag = self.Delta_t_diag[:L, :]
            else:
                # If sequence is longer than pre-initialized, pad with zeros
                padding = np.zeros((L - self.Delta_t_diag.shape[0], self.R))
                Delta_t_diag = np.concatenate([self.Delta_t_diag, padding], axis=0)
            
            Delta_t = jax.vmap(np.diag)(Delta_t_diag)  # (L, R, R)
        else:
            Delta_t = None

        xs, ys = apply_ssm(self.Lambda_bar,
                           self.B_bar,
                           self.C_tilde,
                           input_sequence,
                           self.conj_sym,
                           self.bidirectional,
                           self.E,
                           Delta_t,
                           self.F)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


def init_ExtendedS5SSM(H,
               P,
               R,
               Lambda_re_init,
               Lambda_im_init,
               V,
               Vinv,
               C_init,
               discretization,
               dt_min,
               dt_max,
               conj_sym,
               clip_eigs,
               bidirectional
               ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(ExtendedS5SSM,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   H=H,
                   P=P,
                   R=R,
                   C_init=C_init,
                   discretization=discretization,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional)