from flax import linen as nn
import jax
import jax.numpy as np
from jax.nn.initializers import normal
from .ssm import S5SSM, apply_ssm
from functools import partial
from dataclasses import field

class ExtendedS5SSM(nn.Module):
    original_ssm: nn.Module
    H: int  # input dim
    P: int  # state dim
    R: int  # projection dim
    ssm_kwargs: dict = field(default_factory=dict)
    step_rescale: float = 1.0

    def setup(self):
        kwargs = dict(self.ssm_kwargs)
        kwargs['step_rescale'] = self.step_rescale
        # 기존 S5SSM 불러오기
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

        # p_k 생성용 MLP: input H → hidden → output R
        self.mlp_dense1 = nn.Dense(features=self.R * 2, kernel_init=nn.initializers.normal(0.01))
        self.layer_norm = nn.LayerNorm()
        self.mlp_dense2 = nn.Dense(features=self.R, kernel_init=nn.initializers.normal(0.01))

        # B 확장용 학습 파라미터 E: shape (P, R)
        self.E = self.param("E", nn.initializers.normal(0.01), (self.P, self.R))

        # output projection
        self.G = self.param("G", nn.initializers.normal(0.01), (self.H, self.R))  # for r_k
        self.D = self.param("D", nn.initializers.zeros, (self.H, self.H))  # optional

    def __call__(self, input_sequence):
        """
        input_sequence: (L, H) — 시퀀스 길이 L, 입력 차원 H
        """
        # Step 1: Precompute p_k = MLP(u_k) outside recurrence
        def mlp_fn(u):
            h = self.layer_norm(nn.gelu(self.mlp_dense1(u)))
            p = self.mlp_dense2(h)            # (R,)
            # return np.tanh(p) * 0.1           # scale down to prevent explosion
            return p
        p_seq = jax.vmap(mlp_fn)(input_sequence)  # (L, R)

        # Step 2: Concatenate [u_k; p_k]
        up_seq = np.concatenate([input_sequence, p_seq], axis=-1)  # (L, H+R)

        # Step 3: Construct extended B matrix
        B_ext = np.concatenate([self.ssm.B_bar, self.E], axis=1)  # (P, H+R)

        # Step 4: SSM 계산
        x_seq, _ = apply_ssm(
            self.ssm.Lambda_bar,
            B_ext,
            None,  # no projection matrix → return state
            up_seq,
            conj_sym=self.ssm.conj_sym,
            bidirectional=self.ssm.bidirectional
        )

        # Step 5: Output = C x + G p + D u
        if self.ssm.conj_sym:
            y_seq = jax.vmap(lambda x, p, u: 2 * (self.ssm.C_tilde @ x).real + (self.G @ p) + (self.D @ u))(
                x_seq, p_seq, input_sequence)
        else:
            y_seq = jax.vmap(lambda x, p, u: (self.ssm.C_tilde @ x).real + (self.G @ p) + (self.D @ u))(
                x_seq, p_seq, input_sequence)

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