from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal

"""
Extended S5 with optional auxiliary state p_t and time-dependent scaling Δ(t).
Two auxiliary modes:
  - "absorbed":    use p_t = Δ(t-1) x_t and absorb EΔ into Λ_t so we still scan x only
  - "explicit":    keep s_t = [x_t, p_t] and scan 2x2 block dynamics per mode

Notes
-----
* We keep the original S5 parameterization (Λ of size P; B/C possibly of size local_P when conj_sym=True).
* E is restricted to be diagonal (vector of length P) for efficiency and numerical stability.
* Δ(t) can be parameterized with several types (linear, exponential, sinusoidal, polynomial, constant),
  optionally bounded to (0,1) via sigmoid.
"""

# =========================
# Discretization functions
# =========================

def discretize_bilinear(Lambda, B_tilde, Delta):
    """Bilinear transform for diagonal SSM.
    Args:
        Lambda: (P,) complex
        B_tilde: (P, H) complex
        Delta: (P,) float
    Returns:
        Lambda_bar: (P,) complex, B_bar: (P, H) complex
    """
    Identity = np.ones(Lambda.shape[0])
    BL = 1.0 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Zero-order hold for diagonal SSM.
    Args:
        Lambda: (P,) complex
        B_tilde: (P, H) complex
        Delta: (P,) float
    Returns:
        Lambda_bar: (P,) complex, B_bar: (P, H) complex
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    # handle small |Lambda| numerically by series via where
    denom = np.where(np.abs(Lambda) < 1e-6, 1.0, Lambda)
    coeff = (Lambda_bar - Identity) / denom
    B_bar = coeff[..., None] * B_tilde
    return Lambda_bar, B_bar


# =========================
# Parallel scan operators
# =========================

@jax.vmap
def _binary_operator(q_i, q_j):
    """Associative operator for x_{t+1} = A_t x_t + b_t (diagonal A).
    q = (A_t, b_t) with shapes (P,), (P,).
    Returns (A_j*A_i, A_j*b_i + b_j).
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def _affine_binop_2x2(q_i, q_j):
    """Associative operator for 2x2 block affine maps: (A,b)◦(A',b')=(AA', Ab'+b).
    A: (2,2), b: (2,).
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j @ A_i, A_j @ b_i + b_j


# =========================
# Time-dependent Δ(t)
# =========================

def compute_time_dependent_delta(L, delta_type="linear", delta_params=None, bound_to_01=False):
    """Compute Δ(t) for t=0..L-1.
    delta_params defaults (shape depends on type):
      linear:      [a0, a1]
      exponential: [a0, a1]   -> a0 * exp(-a1 * t)
      sinusoidal:  [a0, a1, a2, a3] -> a0 + a1*sin(a2*t + a3)
      polynomial:  [a0..a4] (up to 4th order)
      constant:    [c]
    If bound_to_01, apply sigmoid.
    """
    t = np.arange(L, dtype=np.float32)
    tn = t / np.maximum(L, 1.0)

    if delta_params is None:
        # sensible small scale default
        if delta_type == "sinusoidal":
            delta_params = np.array([1.0, 0.1, 2.0*np.pi, 0.0])
        elif delta_type == "polynomial":
            delta_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        elif delta_type == "constant":
            delta_params = np.array([1.0])
        else:  # linear / exponential
            delta_params = np.array([1.0, 0.1])

    if delta_type == "linear":
        delta = delta_params[0] + delta_params[1] * tn
    elif delta_type == "exponential":
        delta = delta_params[0] * np.exp(-delta_params[1] * tn)
    elif delta_type == "sinusoidal":
        delta = delta_params[0] + delta_params[1] * np.sin(delta_params[2] * tn + delta_params[3])
    elif delta_type == "polynomial":
        delta = np.zeros((L,), dtype=np.float32)
        for i, c in enumerate(delta_params):
            delta = delta + c * (tn ** i)
    elif delta_type == "constant":
        delta = np.ones((L,), dtype=np.float32) * delta_params[0]
    else:
        raise ValueError(f"Unknown delta_type: {delta_type}")

    if bound_to_01:
        delta = jax.nn.sigmoid(delta)
    return delta


# =========================
# Core apply functions
# =========================

def _apply_outputs_from_states(states, C_tilde, conj_sym):
    """Map complex states to real outputs via C_tilde.
    states: (L, P) complex or (L, 2P) if bidirectional concat.
    C_tilde: (H, P) or (H, 2P) complex.
    Returns: (L, H) real.
    """
    if conj_sym:
        return jax.vmap(lambda x: 2.0 * (C_tilde @ x).real)(states)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(states)


def apply_ssm_absorbed(Lambda_bar, B_bar, C_tilde, E_diag, input_sequence,
                        delta_params, delta_type, conj_sym, bidirectional,
                        bound_delta):
    """Absorb p_t = Δ(t-1) x_t into Λ'_t = Λ_t + E * Δ(t-1) and scan x only.
    Lambda_bar: (P,) complex or (L,P) complex
    B_bar: (P,H) complex
    C_tilde: (H,P) or (H,2P) complex
    E_diag: (P,) complex (diagonal of E)
    input_sequence: (L,H) float
    Returns: outputs (L,H) real
    """
    L, H = input_sequence.shape
    P = Lambda_bar.shape[-1]

    # build Δ(t-1)
    Delta_t = compute_time_dependent_delta(L, delta_type, delta_params, bound_to_01=bound_delta)
    Delta_prev = np.concatenate([np.array([0.0], dtype=Delta_t.dtype), Delta_t[:-1]])  # (L,)

    # expand Lambda over time
    if Lambda_bar.ndim == 1:
        Lambda_base = np.broadcast_to(Lambda_bar, (L, P))
    else:
        Lambda_base = Lambda_bar  # (L,P)

    # modified diagonal per time step
    Lambda_mod = Lambda_base + (E_diag[None, :] * Delta_prev[:, None])

    # b_t = B_bar @ u_t (complex)
    Bu = jax.vmap(lambda u: B_bar @ u)(input_sequence)  # (L,P)

    # scan for states x_t
    _, xs = jax.lax.associative_scan(_binary_operator, (Lambda_mod, Bu))

    # bidirectional
    if bidirectional:
        # reverse scan on reversed inputs; we just reuse Lambda_mod/Bu reversed
        _, xs_rev = jax.lax.associative_scan(_binary_operator, (Lambda_mod[::-1], Bu[::-1]))
        xs_rev = xs_rev[::-1]
        xs = np.concatenate([xs, xs_rev], axis=-1)  # (L, 2P)

    ys = _apply_outputs_from_states(xs, C_tilde, conj_sym)
    return ys


def apply_ssm_explicit(Lambda_bar, B_bar, C_tilde, E_diag, input_sequence,
                        delta_params, delta_type, conj_sym, bidirectional,
                        bound_delta):
    """Explicitly scan augmented state s_t=[x_t, p_t] with 2x2 blocks per mode.
    Lambda_bar: (P,) or (L,P) complex
    B_bar: (P,H) complex
    E_diag: (P,) complex
    input_sequence: (L,H) float
    Returns: outputs (L,H) real
    """
    L, H = input_sequence.shape
    P = Lambda_bar.shape[-1]

    # Δ(t)
    Delta_t = compute_time_dependent_delta(L, delta_type, delta_params, bound_to_01=bound_delta)  # (L,)

    # expand Lambda over time
    if Lambda_bar.ndim == 1:
        Lambda_base = np.broadcast_to(Lambda_bar, (L, P))
    else:
        Lambda_base = Lambda_bar  # (L,P)

    # Bu per time (complex)
    Bu = jax.vmap(lambda u: B_bar @ u)(input_sequence)  # (L,P)

    # Build per (t,k) 2x2 A and 2-dim b
    lam = Lambda_base[..., None, None]             # (L,P,1,1)
    e = E_diag[None, :, None, None]                # (1,P,1,1)
    dlt = Delta_t[:, None, None, None]             # (L,1,1,1)

    A = np.concatenate([
            np.concatenate([lam,         e], axis=-1),
            np.concatenate([dlt * lam, dlt * e], axis=-1)
        ], axis=-2)                                 # (L,P,2,2)

    b = np.stack([Bu, Delta_t[:, None] * Bu], axis=-1)  # (L,P,2)

    # Scan along time for each mode independently
    def scan_one_mode(Ak, bk):
        # Ak: (L,2,2), bk: (L,2)
        _, s = jax.lax.associative_scan(_affine_binop_2x2, (Ak, bk))  # (L,2)
        return s  # (L,2)

    s_all = jax.vmap(scan_one_mode, in_axes=(1, 1))(A, b)  # (P,L,2)
    s_all = s_all.transpose((1, 0, 2))  # (L,P,2)
    x = s_all[..., 0]

    if bidirectional:
        A_rev = A[::-1]
        b_rev = b[::-1]
        s_all_rev = jax.vmap(scan_one_mode, in_axes=(1, 1))(A_rev, b_rev)  # (P,L,2)
        s_all_rev = s_all_rev.transpose((1, 0, 2))[::-1]
        x_rev = s_all_rev[..., 0]
        states = np.concatenate([x, x_rev], axis=-1)  # (L,2P)
    else:
        states = x  # (L,P)

    ys = _apply_outputs_from_states(states, C_tilde, conj_sym)
    return ys


# =========================
# Extended S5 Module
# =========================

class ExtendedS5SSM(nn.Module):
    """S5 SSM extended with diagonal E and time-dependent Δ(t).

    Args are similar to S5SSM. New ones:
        enable_auxiliary: bool, turn on auxiliary pathway
        aux_mode: "absorbed" | "explicit"
        delta_type: one of {linear, exponential, sinusoidal, polynomial, constant}
        bound_delta: if True, applies sigmoid to Δ(t)
    """

    # Base S5 parameters
    Lambda_re_init: np.DeviceArray
    Lambda_im_init: np.DeviceArray
    V: np.DeviceArray
    Vinv: np.DeviceArray

    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = True
    bidirectional: bool = False
    step_rescale: float = 1.0

    # Auxiliary controls
    enable_auxiliary: bool = True
    aux_mode: str = "absorbed"        # or "explicit"
    delta_type: str = "linear"
    bound_delta: bool = False

    def setup(self):
        # local_P for B/C only (matches original S5 convention)
        local_P = 2 * self.P if self.conj_sym else self.P

        # Λ parameters (size P)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # B (local_P x H) real-pair -> complex
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param(
            "B",
            lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv),
            B_shape,
        )
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]  # (local_P,H) complex

        # C build
        if self.C_init == "trunc_standard_normal":
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init == "lecun_normal":
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init == "complex_normal":
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(f"C_init method {self.C_init} not implemented")

        if self.C_init == "complex_normal":
            Ccols = 2 * self.P if self.bidirectional else self.P
            C = self.param("C", C_init, (self.H, Ccols, 2))
            self.C_tilde = C[..., 0] + 1j * C[..., 1]
        else:
            if self.bidirectional:
                self.C1 = self.param("C1", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
                self.C2 = self.param("C2", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate([C1, C2], axis=-1)  # (H, 2*local_P)
            else:
                self.C = self.param("C", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Feedthrough D
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Learnable timescales (size P; follows original S5 pattern)
        self.log_step = self.param("log_step", init_log_steps, (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])  # (P,)

        # Discretize
        if self.discretization == "zoh":
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization == "bilinear":
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError(f"Discretization method {self.discretization} not implemented")

        # Auxiliary params
        if self.enable_auxiliary:
            # diagonal E over P modes (complex via real/imag)
            self.E_diag_real = self.param("E_diag_real", normal(stddev=0.05), (self.P,))
            self.E_diag_imag = self.param("E_diag_imag", normal(stddev=0.05), (self.P,))
            self.E_diag = self.E_diag_real + 1j * self.E_diag_imag

            # Δ params (size depends on delta_type)
            if self.delta_type == "polynomial":
                n_params = 5
            elif self.delta_type == "sinusoidal":
                n_params = 4
            elif self.delta_type == "constant":
                n_params = 1
            else:  # linear, exponential
                n_params = 2
            self.delta_params = self.param("delta_params", normal(stddev=0.1), (n_params,))

    def __call__(self, input_sequence):
        """Forward: returns (L,H) real outputs.
        input_sequence: (L,H) float
        """
        if self.enable_auxiliary:
            if self.aux_mode == "explicit":
                ys = apply_ssm_explicit(
                    self.Lambda_bar, self.B_bar, self.C_tilde, self.E_diag,
                    input_sequence, self.delta_params, self.delta_type,
                    self.conj_sym, self.bidirectional, self.bound_delta,
                )
            elif self.aux_mode == "absorbed":
                ys = apply_ssm_absorbed(
                    self.Lambda_bar, self.B_bar, self.C_tilde, self.E_diag,
                    input_sequence, self.delta_params, self.delta_type,
                    self.conj_sym, self.bidirectional, self.bound_delta,
                )
            else:
                raise ValueError(f"Unknown aux_mode: {self.aux_mode}")
        else:
            # Standard S5 scan without auxiliary
            L = input_sequence.shape[0]
            P = self.Lambda_bar.shape[0]
            Lambda_elements = np.broadcast_to(self.Lambda_bar, (L, P))
            Bu = jax.vmap(lambda u: self.B_bar @ u)(input_sequence)
            _, xs = jax.lax.associative_scan(_binary_operator, (Lambda_elements, Bu))
            if self.bidirectional:
                _, xs_rev = jax.lax.associative_scan(_binary_operator, (Lambda_elements[::-1], Bu[::-1]))
                xs = np.concatenate([xs, xs_rev[::-1]], axis=-1)
            ys = _apply_outputs_from_states(xs, self.C_tilde, self.conj_sym)

        # add feedthrough Du
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


# =========================
# Convenience initializer
# =========================

def init_ExtendedS5SSM(
    H,
    P,
    Lambda_re_init,
    Lambda_im_init,
    V,
    Vinv,
    C_init="lecun_normal",
    discretization="bilinear",
    dt_min=1e-3,
    dt_max=1e-1,
    conj_sym=True,
    clip_eigs=True,
    bidirectional=False,
    enable_auxiliary=True,
    aux_mode="absorbed",
    delta_type="linear",
    bound_delta=False,
):
    """Return a partial to construct ExtendedS5SSM with fixed hyper-params."""
    return partial(
        ExtendedS5SSM,
        H=H,
        P=P,
        Lambda_re_init=Lambda_re_init,
        Lambda_im_init=Lambda_im_init,
        V=V,
        Vinv=Vinv,
        C_init=C_init,
        discretization=discretization,
        dt_min=dt_min,
        dt_max=dt_max,
        conj_sym=conj_sym,
        clip_eigs=clip_eigs,
        bidirectional=bidirectional,
        enable_auxiliary=enable_auxiliary,
        aux_mode=aux_mode,
        delta_type=delta_type,
        bound_delta=bound_delta,
    )
