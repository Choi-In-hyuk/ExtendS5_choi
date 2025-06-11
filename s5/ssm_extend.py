from flax import linen as nn
import jax
import jax.numpy as np
from jax.nn.initializers import normal
from .ssm import S5SSM, apply_ssm
from functools import partial
from dataclasses import field


class ExtendedS5SSM(nn.Module):
    original_ssm: S5SSM
    P: int
    H: int
    R: int
    ssm_kwargs: dict = field(default_factory=dict)
    step_rescale: float = 1.0
    conj_sym: bool = False

    def setup(self):
        kwargs = dict(self.ssm_kwargs)
        kwargs['step_rescale'] = self.step_rescale

        self.ssm = S5SSM(
            H=self.original_ssm.H,
            P=self.original_ssm.P,
            Lambda_re_init=self.original_ssm.Lambda_re_init,
            Lambda_im_init=self.original_ssm.Lambda_im_init,
            V=self.original_ssm.V,
            Vinv=self.original_ssm.Vinv,
            C_init=self.original_ssm.C_init,
            discretization=self.original_ssm.discretization,
            dt_min=self.original_ssm.dt_min,
            dt_max=self.original_ssm.dt_max,
            conj_sym=self.conj_sym,
            clip_eigs=self.original_ssm.clip_eigs,
            bidirectional=self.original_ssm.bidirectional,
            step_rescale=self.step_rescale
        )

        self.F = self.param("F", normal(stddev=0.1), (self.R, self.P))
        self.E = self.param("E", normal(stddev=0.01), (self.P, self.R))
        self.Delta = self.param("Delta", normal(stddev=1.0), (self.R, self.R))

    def __call__(self, input_sequence):
        Lambda_bar = self.ssm.Lambda_bar
        B_bar = self.ssm.B_bar
        C_tilde = self.ssm.C_tilde
        D = self.ssm.D

        L = input_sequence.shape[0]

        # 1st pass: compute x_seq with standard SSM (no p_k)
        x_seq = apply_ssm(Lambda_bar, B_bar, np.eye(self.P), input_sequence, conj_sym=False, bidirectional=False)

        # Compute q_k = F x_{k-1}, use x_seq[:-1]
        x_prev = x_seq[:-1]  # shape (L-1, P)
        q_seq = jax.vmap(lambda x: self.F @ x)(x_prev)  # (L-1, R)
        p_seq = q_seq @ self.Delta    # (L-1, R)

        '''
        jax.debug.print("x_seq.max: {}", x_seq.max())
        jax.debug.print("p_seq.max: {}", p_seq.max())
        '''
        # Truncate u to match (L-1, H), and concat with p to get [u; p]
        u_trunc = input_sequence[1:]                    # (L-1, H)
        up_seq = np.concatenate([u_trunc, p_seq], axis=-1)  # (L-1, H+R)

        # Prepare extended B matrix: [B E] ∈ (P, H+R)
        B_ext = np.concatenate([B_bar, self.E], axis=1)  # (P, H+R)
        
        '''
        jax.debug.print("B_bar.max: {}", B_bar.max())
        jax.debug.print("E.max: {}", self.E.max())
        '''
        # 2nd pass: compute new x'_seq with augmented input
        x_aug = apply_ssm(Lambda_bar, B_ext, C_tilde, up_seq, conj_sym=self.conj_sym, bidirectional=self.ssm.bidirectional)

        # pad to L length
        zero = np.zeros((1, x_aug.shape[1]), dtype=x_aug.dtype)
        x_aug_padded = np.concatenate([zero, x_aug], axis=0)  # (L, H)

        # Du도 같은 방식으로 padding
        Du_trunc = jax.vmap(lambda u: D * u)(u_trunc)  # (L-1, H)
        zero_D = np.zeros((1, Du_trunc.shape[1]), dtype=Du_trunc.dtype)
        Du = np.concatenate([zero_D, Du_trunc], axis=0)  # (L, H)

        return x_aug_padded + Du  # (L, H)


def init_ExtendedS5SSM(original_ssm, P, H, R, ssm_kwargs=None):
    if ssm_kwargs is None:
        ssm_kwargs = {}
    return partial(ExtendedS5SSM,
                   original_ssm=original_ssm,
                   P=P,
                   H=H,
                   R=R,
                   ssm_kwargs=ssm_kwargs)
