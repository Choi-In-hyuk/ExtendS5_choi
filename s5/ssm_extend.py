# 개선된 보조 상태를 포함한 S5 SSM 구현

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from functools import partial

# 기존 discretization 함수들은 동일하게 유지
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Bilinear discretization method"""
    Identity = jnp.ones(Lambda.shape[0])
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar

def discretize_zoh(Lambda, B_tilde, Delta):
    """Zero-order hold discretization method"""
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar

@jax.vmap
def binary_operator(q_i, q_j):
    """Standard binary operator for parallel scan"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

@jax.vmap
def binary_operator_with_auxiliary_correct(q_i, q_j):
    """
    올바른 보조 상태를 포함한 binary operator
    
    업데이트 방정식:
    x_{t+1} = A_t * x_t + B_t * u_t + E * p_t
    p_{t+1} = Δ(t) * x_{t+1}
    
    여기서 p_t는 이전 단계의 x_t에 시간 스케일링을 적용한 것
    
    Args:
        q_i, q_j: (A, Bu, Ep, Delta_next) tuples
    """
    A_i, Bu_i, Ep_i, Delta_i = q_i
    A_j, Bu_j, Ep_j, Delta_j = q_j
    
    # 표준 SSM 업데이트 with auxiliary term
    A_out = A_j * A_i
    Bu_out = A_j * Bu_i + Bu_j
    Ep_out = A_j * Ep_i + Ep_j
    
    # Delta는 시간 의존적이므로 각 단계별로 적용
    # 실제 p 업데이트는 scan 후에 별도로 처리
    Delta_out = Delta_j  # 현재 시간 스텝의 Delta
    
    return A_out, Bu_out, Ep_out, Delta_out

def compute_time_dependent_delta(sequence_length, delta_type="linear", delta_params=None):
    """
    시간 의존적 스케일링 함수 Δ(t) 계산
    
    Args:
        sequence_length: 시퀀스 길이
        delta_type: 스케일링 타입
        delta_params: 스케일링 파라미터
    
    Returns:
        Delta_t: 시간 의존적 스케일링 (L,)
    """
    time_indices = jnp.arange(sequence_length, dtype=jnp.float32)
    t_norm = time_indices / sequence_length  # [0, 1] 정규화
    
    if delta_params is None:
        delta_params = jnp.array([1.0, 0.1, 0.0, 0.0, 0.0])
    
    if delta_type == "linear":
        return delta_params[0] + delta_params[1] * t_norm
    
    elif delta_type == "exponential":
        return delta_params[0] * jnp.exp(-delta_params[1] * t_norm)
    
    elif delta_type == "sinusoidal":
        return delta_params[0] + delta_params[1] * jnp.sin(delta_params[2] * t_norm + delta_params[3])
    
    elif delta_type == "polynomial":
        # 다항식: a0 + a1*t + a2*t² + a3*t³ + a4*t⁴
        delta = jnp.zeros(sequence_length)
        for i, coeff in enumerate(delta_params):
            delta += coeff * (t_norm ** i)
        return delta
    
    else:  # constant
        return delta_params[0] * jnp.ones(sequence_length)

def apply_ssm_with_auxiliary_correct(Lambda_bar, B_bar, C_tilde, E_diag, 
                                   input_sequence, delta_params, delta_type, 
                                   conj_sym, bidirectional):
    """
    올바른 보조 상태 업데이트를 포함한 SSM 적용
    
    수학적 관계:
    x_{t+1} = A_t * x_t + B_t * u_t + E * p_t
    p_{t+1} = Δ(t) * x_{t+1}
    
    Args:
        Lambda_bar: 이산화된 상태 행렬 (P,) or (L, P)
        B_bar: 이산화된 입력 행렬 (P, H)  
        C_tilde: 출력 행렬 (H, P)
        E_diag: 보조 상태 영향 대각 벡터 (P,)
        input_sequence: 입력 시퀀스 (L, H)
        delta_params: Δ(t) 파라미터
        delta_type: Δ(t) 타입
        conj_sym: conjugate symmetry 여부
        bidirectional: 양방향 여부
    
    Returns:
        states, outputs: 상태와 출력
    """
    L, H = input_sequence.shape
    P = Lambda_bar.shape[-1]
    
    # Lambda_bar 확장
    if len(Lambda_bar.shape) == 1:
        Lambda_elements = Lambda_bar * jnp.ones((L, P))
    else:
        Lambda_elements = Lambda_bar
    
    # 입력 요소 계산
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)
    
    # 시간 의존적 스케일링 계산
    Delta_t = compute_time_dependent_delta(L, delta_type, delta_params)
    
    # 초기 p_0 = 0으로 설정
    p_init = jnp.zeros(P, dtype=jnp.complex64)
    
    # 시간 단계별로 순차적으로 업데이트
    def scan_fn(carry, inputs):
        x_prev, p_prev = carry
        Lambda_t, Bu_t, Delta_t_current = inputs
        
        # x_{t+1} = Λ_t * x_t + Bu_t + E * p_t
        Ep_t = E_diag * p_prev  # element-wise multiplication
        x_next = Lambda_t * x_prev + Bu_t + Ep_t
        
        # p_{t+1} = Δ(t) * x_{t+1}  
        p_next = Delta_t_current * x_next
        
        return (x_next, p_next), x_next
    
    # 초기 상태
    x_init = jnp.zeros(P, dtype=jnp.complex64)
    
    # 순차 스캔 실행
    _, states = jax.lax.scan(
        scan_fn,
        (x_init, p_init),
        (Lambda_elements, Bu_elements, Delta_t)
    )
    
    # 양방향 처리
    if bidirectional:
        # 역방향 스캔
        _, states_rev = jax.lax.scan(
            scan_fn,
            (x_init, p_init), 
            (Lambda_elements[::-1], Bu_elements[::-1], Delta_t[::-1])
        )
        states_rev = states_rev[::-1]  # 순서 복원
        states = jnp.concatenate([states, states_rev], axis=-1)
    
    # 출력 계산
    if C_tilde is None:
        return states, None
    
    if conj_sym:
        outputs = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(states)
    else:
        outputs = jax.vmap(lambda x: (C_tilde @ x).real)(states)
    
    return states, outputs


class S5SSMWithAuxiliaryImproved(nn.Module):
    """개선된 보조 상태를 포함한 S5 SSM"""
    
    # 기본 S5 파라미터들
    Lambda_re_init: jax.Array
    Lambda_im_init: jax.Array
    V: jax.Array
    Vinv: jax.Array
    H: int
    P: int
    C_init: str = "lecun_normal"
    discretization: str = "bilinear"
    dt_min: float = 0.001
    dt_max: float = 0.1
    conj_sym: bool = True
    clip_eigs: bool = True
    bidirectional: bool = False
    step_rescale: float = 1.0
    
    # 보조 상태 파라미터들
    delta_type: str = "linear"  # "linear", "exponential", "sinusoidal", "polynomial", "constant"
    enable_auxiliary: bool = True
    
    def setup(self):
        """파라미터 초기화"""
        
        if self.conj_sym:
            local_P = 2 * self.P
        else:
            local_P = self.P
        
        # Lambda 초기화
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        
        if self.clip_eigs:
            self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im
        
        # B 행렬 초기화
        B_init = lecun_normal()
        self.B = self.param("B", B_init, (local_P, self.H, 2))
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        
        # C 행렬 초기화
        if self.C_init == "lecun_normal":
            C_init = lecun_normal()
        else:
            C_init = normal(stddev=0.5**0.5)
        
        if self.bidirectional:
            self.C = self.param("C", C_init, (self.H, 2 * local_P, 2))
        else:
            self.C = self.param("C", C_init, (self.H, local_P, 2))
        
        self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]
        
        # D 행렬 (feedthrough)
        self.D = self.param("D", normal(stddev=1.0), (self.H,))
        
        # 보조 상태 관련 파라미터
        if self.enable_auxiliary:
            # E를 대각 벡터로 초기화 (P,) - 파라미터 수 대폭 감소
            self.E_diag_real = self.param("E_diag_real", normal(stddev=0.1), (local_P,))
            self.E_diag_imag = self.param("E_diag_imag", normal(stddev=0.1), (local_P,))
            self.E_diag = self.E_diag_real + 1j * self.E_diag_imag
            
            # 시간 의존적 스케일링 Δ(t) 파라미터들
            if self.delta_type == "polynomial":
                n_params = 5  # 4차 다항식
            elif self.delta_type == "sinusoidal":
                n_params = 4  # amplitude, frequency, phase, offset
            else:
                n_params = 2  # 대부분의 경우 2개 파라미터로 충분
                
            self.delta_params = self.param("delta_params", 
                                         normal(stddev=0.1), 
                                         (n_params,))
        
        # 시간 스케일 초기화
        self.log_step = self.param("log_step", 
                                  lambda rng, shape: jnp.log(jax.random.uniform(rng, shape, 
                                                                               minval=self.dt_min, 
                                                                               maxval=self.dt_max)), 
                                  (self.P,))
        
        step = self.step_rescale * jnp.exp(self.log_step)
        
        # 이산화
        if self.discretization == "zoh":
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization == "bilinear":
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError(f"Discretization {self.discretization} not implemented")
    
    def __call__(self, input_sequence):
        """순전파"""
        
        if self.enable_auxiliary:
            # 올바른 보조 상태 업데이트를 포함한 SSM 적용
            states, outputs = apply_ssm_with_auxiliary_correct(
                self.Lambda_bar,
                self.B_bar, 
                self.C_tilde,
                self.E_diag,
                input_sequence,
                self.delta_params,
                self.delta_type,
                self.conj_sym,
                self.bidirectional
            )
        else:
            # 표준 SSM 적용
            Lambda_elements = self.Lambda_bar * jnp.ones((input_sequence.shape[0], self.Lambda_bar.shape[0]))
            Bu_elements = jax.vmap(lambda u: self.B_bar @ u)(input_sequence)
            
            _, states = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
            
            if self.conj_sym:
                outputs = jax.vmap(lambda x: 2 * (self.C_tilde @ x).real)(states)
            else:
                outputs = jax.vmap(lambda x: (self.C_tilde @ x).real)(states)
        
        # Feedthrough 추가
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        
        return outputs + Du


def init_S5SSMWithAuxiliaryState(**kwargs):
    """초기화 함수"""
    return partial(S5SSMWithAuxiliaryImproved, **kwargs)