
## 1. 환경 설정

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

## 2. 데이터 다운로드

#### LRA 데이터셋 다운로드
```bash
./bin/download_lra.sh
```

#### 모든 데이터셋 다운로드
```bash
./bin/download_all.sh
```

## 3. 기본 실행

#### cifar10 분류 (보조 상태 비활성화)
```bash
python -m s5.train_ex --dataset lra-cifar-classification --epochs 20 --bsz 32
```
#### 보조 상태 활성화 (absorbed 모드, 선형 스케일링)
```bash
python -m s5.train_ex --dataset lra-cifar-classification --enable_auxiliary --aux_mode absorbed --delta_type linear --epochs 20 --bsz 32
```
#### explicit 모드, 지수 스케일링
```bash
python -m s5.train_ex --dataset lra-cifar-classification --enable_auxiliary --aux_mode explicit --delta_type linear --epochs 20 --bsz 32
```
#### explicit 모드, 사인파 스케일링
```bash
python -m s5.train_ex --dataset lra-cifar-classification --enable_auxiliary --aux_mode explicit --delta_type sinusoidal --epochs 20 --bsz 32
```

## 4. argparse
#### 데이터셋 종류
```bash
--dataset [mnist-classification, lra-cifar-classification, imdb-classification, litsops-classification, pathfinder-classification]
```
#### aux_mode
```bash
--aux_mode [absorbed, explicit]
```
#### Δ(t) type
```bash
--delta_type [linear, exponential, sinusoidal, polynomial, constant]
```

## 추가 설명
### aux mode
<img width="600" height="746" alt="image" src="https://github.com/user-attachments/assets/447cd98b-94d1-42f1-a33c-98cf2cd0b680" />


## Δ(t) type
<img width="610" height="740" alt="image" src="https://github.com/user-attachments/assets/e018ccb3-9874-4154-b60c-3da9c31ea10a" />

