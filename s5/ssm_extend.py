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
            conj_sym=self.original_ssm.conj_sym,
            clip_eigs=self.original_ssm.clip_eigs,
            bidirectional=self.original_ssm.bidirectional,
            step_rescale=self.step_rescale
        )
        self.F = self.param("F", normal(stddev=0.01), (self.R, self.H))
        self.H_proj = self.param("H_proj", normal(stddev=0.01), (self.R, self.H))

        self.Delta_p = self.param("Delta_p", lambda rng, shape: np.eye(self.R), (self.R, self.R))
        self.Delta_r = self.param("Delta_r", lambda rng, shape: np.eye(self.R), (self.R, self.R))

        self.E = self.param("E", normal(stddev=0.01), (self.P, self.R))
        self.G = self.param("G", normal(stddev=0.01), (self.H, self.R))

    def __call__(self, input_sequence):
        """
        input_sequence: (L, H) — 원래 입력 시퀀스 u_k
        output_sequence: (L, H) — 출력 시퀀스 y_k
        """

        # Step 1: Compute p_k and r_k from u_k
        # p_k = Δ F u_k,   r_k = Δ H u_k
        p_seq = jax.vmap(lambda u: self.Delta_p @ (self.F @ u))(input_sequence)  # (L, R)
        r_seq = jax.vmap(lambda u: self.Delta_r @ (self.H_proj @ u))(input_sequence)  # (L, R)

        # Step 2: Form [u_k; p_k]
        up_seq = np.concatenate([input_sequence, p_seq], axis=-1)  # (L, H+R)

        # Step 3: Input projection B_ext = [B E]
        B_ext = np.concatenate([self.ssm.B_bar, self.E], axis=1)  # (P, H+R)

        # Step 4: Compute x_seq with apply_ssm (standard)
        x_seq, _ = apply_ssm(
            self.ssm.Lambda_bar,
            B_ext,
            # np.eye(2*self.ssm.Lambda_bar.shape[0] if self.ssm.conj_sym else self.ssm.Lambda_bar.shape[0]),  # Identity projection: return x_seq
            None,
            up_seq,
            conj_sym=self.original_ssm.conj_sym,
            bidirectional=self.original_ssm.bidirectional
        )

        if self.original_ssm.conj_sym:
            y_seq = jax.vmap(lambda x, r, u: 2*(self.ssm.C_tilde @ x).real + (self.G @ r) + (self.ssm.D @ u))(
                x_seq, r_seq, input_sequence
            )  # (L, H)
        else:
            # Step 5: Compute output y_k = C x_k + G r_k + D u_k
            y_seq = jax.vmap(lambda x, r, u: (self.ssm.C_tilde @ x).real + (self.G @ r) + (self.ssm.D @ u))(
                x_seq, r_seq, input_sequence
            )  # (L, H)

        return y_seq

'''
    if conj_sym:
        return xs, jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return xs, jax.vmap(lambda x: (C_tilde @ x).real)(xs)
        '''


def init_ExtendedS5SSM(original_ssm, P, H, R, ssm_kwargs=None):
    if ssm_kwargs is None:
        ssm_kwargs = {}
    return partial(ExtendedS5SSM,
                   original_ssm=original_ssm,
                   P=P,
                   H=H,
                   R=R,
                   ssm_kwargs=ssm_kwargs)