# ExtendS5 환경 세팅 (JAX 0.3.25 + CUDA 11 계열)

> 목적: Python 3.9 + JAX 0.3.25(+cuda11.cudnn82) + Flax 0.5.3 조합에서 발생하는
> `numpy/scipy` ABI·CUDA 런타임 이슈를 피하면서 CIFAR 실험을 재현.

## 요구사항

- OS: Linux x86_64
- **Python: 3.9**
- NVIDIA Driver: CUDA 11.x 호환(대부분의 최신 드라이버는 OK)
- CUDA 런타임: **11.x** (conda `cudatoolkit`로 제공)

## TL;DR (권장 순서)

새 환경(예:`s5`)에서 진행:

```bash
# 0) conda env (Python 3.9)
conda create -n s5 python=3.9 -y
conda activate s5

# 1) 잔존 패키지 제거 (새로운 환경이면 skip)
pip uninstall -y jax jaxlib numpy scipy flax chex optax || true

# 2) CUDA 11 런타임 설치 (libcudart.so.11.0 제공)
conda install -y -c conda-forge cudatoolkit=11.3

# 3) Numpy/Scipy ABI 고정 (순서 중요: numpy 먼저)
pip install "numpy==1.23.5"
pip install "scipy==1.8.1"

# 4) JAX/Flax 스택
pip install -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  "jax==0.3.25" "jaxlib==0.3.25+cuda11.cudnn82"
pip install "flax==0.5.3" "chex==0.1.5" "optax==0.1.3"

# 5) PyTorch CU113
pip install -f https://download.pytorch.org/whl/cu113/torch_stable.html \
  "torch==1.10.0+cu113" "torchvision==0.11.1+cu113" \
  "torchaudio==0.10.0+cu113" "torchtext==0.11.0"

# 6) 검증
python - <<'PY'
import jax, numpy, scipy, flax, chex, optax, torch
print("JAX:", jax.__version__)
print("NumPy:", numpy.__version__, "SciPy:", scipy.__version__)
print("Flax:", flax.__version__, "Chex:", chex.__version__, "Optax:", optax.__version__)
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("JAX devices:", jax.devices())
PY
```
### wandb 설정
```bash
pip install wandb
```

### einops 설치
```bash
pip install einops
```

### 실행 테스트
```bash
python -m s5.train_ex --dataset lra-cifar-classification --epochs 20 --bsz 32
```

# 데이터 다운로드

#### LRA 데이터셋 다운로드
```bash
./bin/download_lra.sh
```

#### 모든 데이터셋 다운로드
```bash
./bin/download_all.sh
```

# Extended S5 SSM

Extended S5 SSM은 기존 S5 모델에 Lambda extension 기능을 추가한 확장 버전입니다. 이 기능은 시간에 따라 변화하는 상태 행렬을 학습할 수 있게 합니다.

## Lambda Extension

Lambda extension은 다음과 같은 형태로 구현됩니다:

```
Lambda_ex(t) = Lambda_bar + E @ Delta_t(t) @ F
```

여기서:
- `Lambda_bar`: 기본 상태 행렬 (고정)
- `E`: (P, R) 크기의 학습 가능한 행렬
- `Delta_t(t)`: 시간 t에 따른 (R, R) 크기의 대각 행렬
- `F`: (R, P) 크기의 학습 가능한 행렬

## 사용법

### Extended S5 SSM 실행
```bash
python -m s5.train_ex --ssm_type extend --dataset lra-cifar-classification --R 10
```

### 매개변수 설명
- `--ssm_type extend`: Extended S5 SSM 사용 (Lambda extension 자동 활성화)
- `--R`: Delta_t 행렬의 랭크 (기본값: 10)

### 예시 실행
```bash
# 기본 Extended S5 SSM
python -m s5.train_ex --ssm_type extend --dataset lra-cifar-classification --R 10

# 다른 데이터셋에서 사용
python -m s5.train_ex --ssm_type extend --dataset imdb-classification --R 8
python -m s5.train_ex --ssm_type extend --dataset listops-classification --R 12
```

## 성능

Extended S5 SSM은 기존 S5 모델에 비해 다음과 같은 장점을 제공합니다:

- **시간적 적응성**: 시퀀스의 시간적 특성에 따라 상태 행렬이 동적으로 조정됩니다
- **표현력 향상**: 더 복잡한 시퀀스 패턴을 학습할 수 있습니다
- **유연성**: 다양한 데이터셋에서 더 나은 성능을 보일 수 있습니다

### 예시 결과 (CIFAR 분류)
- **최종 검증 정확도**: 69.78%
- **최종 테스트 정확도**: 69.01%
- **훈련 손실**: 0.7034
- **검증 손실**: 0.9053