import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from datasets import load_dataset
from typing import Any, Callable, Sequence
import numpy as np
from s5.ssm import S5SSM, init_S5SSM
from torch.utils.tensorboard import SummaryWriter
import sys
import os

# 데이터셋 로드 함수
def load_dataset_from_json(train_path, val_path, test_path):
    train_data = load_dataset("json", data_files=train_path)["train"]
    val_data = load_dataset("json", data_files=val_path)["train"]
    test_data = load_dataset("json", data_files=test_path)["train"]
    return train_data, val_data, test_data

# S5 기반 선택적 복사 모델
class SelectiveCopyingS5(nn.Module):
    vocab_size: int
    d_model: int
    ssm_size: int
    ssm_init: dict
    input_length: int = 4096
    output_length: int = 16

    def setup(self):
        self.ssm = init_S5SSM(**self.ssm_init)
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # 임베딩 레이어
        x = nn.Embed(self.vocab_size, self.d_model)(x)  # (B, L_in, D)
        
        # S5 레이어
        ssm = init_S5SSM(**self.ssm_init)
        x = ssm(x)  # (B, L_in, D)
        
        # 출력 시퀀스 길이에 맞게 조정
        # 마지막 output_length 개의 타임스텝만 사용
        x = x[:, -self.output_length:, :]  # (B, L_out, D)
        
        # 출력 레이어
        x = nn.Dense(self.d_model)(x)
        x = nn.relu(x)
        x = nn.Dense(self.vocab_size)(x)  # (B, L_out, vocab_size)
        return x

# 학습 상태 클래스
class TrainState(train_state.TrainState):
    key: jax.random.PRNGKey

# 학습 스텝 함수
@jax.jit
def train_step(state: TrainState, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits.reshape(-1, logits.shape[-1]),
            labels=batch['output'].reshape(-1)
        ).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    # 정확도 계산
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['output'])
    
    return state, loss, accuracy

# 검증 스텝 함수
@jax.jit
def eval_step(state: TrainState, batch):
    logits = state.apply_fn({'params': state.params}, batch['input'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.reshape(-1, logits.shape[-1]),
        labels=batch['output'].reshape(-1)
    ).mean()
    
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['output'])
    
    return loss, accuracy

def main():
    # 하이퍼파라미터 설정
    config = {
        'vocab_size': 16,
        'd_model': 512,
        'ssm_size': 64,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'num_epochs': 5,
        'seed': 42,
        'input_length': 4096,
        'output_length': 16
    }
    
    # 데이터 로드
    train_data, val_data, test_data = load_dataset_from_json(
        "/workspace/S52/project/selective_copying_data/train.jsonl",
        "/workspace/S52/project/selective_copying_data/validation.jsonl",
        "/workspace/S52/project/selective_copying_data/test.jsonl"
    )
    
    # S5 초기화 파라미터
    ssm_init = {
        'Lambda_re_init': jnp.zeros(config['ssm_size'], dtype=jnp.float32),
        'Lambda_im_init': jnp.zeros(config['ssm_size'], dtype=jnp.float32),
        'V': jnp.eye(config['ssm_size'], dtype=jnp.float32),
        'Vinv': jnp.eye(config['ssm_size'], dtype=jnp.float32),
        'C_init': "trunc_standard_normal",
        'discretization': "zoh",
        'dt_min': 0.001,
        'dt_max': 0.1,
        'conj_sym': False,
        'clip_eigs': True,
        'bidirectional': False
    }
    
    # 모델 초기화
    model = SelectiveCopyingS5(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        ssm_size=config['ssm_size'],
        ssm_init=ssm_init,
        input_length=config['input_length'],
        output_length=config['output_length']
    )
    
    # 옵티마이저 설정
    optimizer = optax.adam(config['learning_rate'])
    
    # 초기 상태 생성
    key = jax.random.PRNGKey(config['seed'])
    key, init_key = jax.random.split(key)
    
    # 더미 입력으로 모델 초기화
    dummy_input = jnp.ones((1, config['input_length']), dtype=jnp.int32)
    variables = model.init(init_key, dummy_input)
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        key=key
    )
    
    # 학습 루프
    for epoch in range(config['num_epochs']):
        # 학습
        train_losses = []
        train_accuracies = []
        
        for batch in train_data:
            batch = {
                'input': jnp.array(batch['input']),
                'output': jnp.array(batch['output'])
            }
            state, loss, accuracy = train_step(state, batch)
            train_losses.append(loss)
            train_accuracies.append(accuracy)
        
        avg_train_loss = jnp.mean(jnp.array(train_losses))
        avg_train_acc = jnp.mean(jnp.array(train_accuracies))
        
        # 검증
        val_losses = []
        val_accuracies = []
        
        for batch in val_data:
            batch = {
                'input': jnp.array(batch['input']),
                'output': jnp.array(batch['output'])
            }
            loss, accuracy = eval_step(state, batch)
            val_losses.append(loss)
            val_accuracies.append(accuracy)
        
        avg_val_loss = jnp.mean(jnp.array(val_losses))
        avg_val_acc = jnp.mean(jnp.array(val_accuracies))
        
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 