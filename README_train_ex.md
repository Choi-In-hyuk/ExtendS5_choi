# S5SSMWithAuxiliaryState ListOps 학습 가이드

이 가이드는 `S5SSMWithAuxiliaryState`를 사용하여 LRA (Long Range Arena)의 ListOps 태스크를 학습시키는 방법을 설명합니다.

## 📋 개요

`S5SSMWithAuxiliaryState`는 기존 S5 SSM에 보조 상태 `p_t`를 추가한 확장 버전입니다:

```
x_{t+1} = A x_t + B u_t + E p_t
p_{t+1} = Δ(t) * x_{t+1}
```

## 🚀 빠른 시작

### 1. 단일 실험 실행

```bash
# 기본 S5SSM (보조 상태 비활성화)
python train_ex.py --epochs 20 --bsz 32

# 보조 상태 활성화 (선형 스케일링)
python train_ex.py --enable_auxiliary_state --time_scale_type linear --epochs 20 --bsz 32

# 보조 상태 활성화 (지수 스케일링)
python train_ex.py --enable_auxiliary_state --time_scale_type exponential --epochs 20 --bsz 32

# 보조 상태 활성화 (사인파 스케일링)
python train_ex.py --enable_auxiliary_state --time_scale_type sinusoidal --epochs 20 --bsz 32
```

### 2. 배치 실험 실행

```bash
# 모든 실험을 순차적으로 실행
./run_train_ex.sh
```

## ⚙️ 주요 파라미터

### 보조 상태 관련 파라미터

| 파라미터 | 설명 | 기본값 | 옵션 |
|---------|------|--------|------|
| `--enable_auxiliary_state` | 보조 상태 활성화 | False | True/False |
| `--auxiliary_strength` | 보조 상태 영향 강도 | 0.1 | float |
| `--time_scale_type` | 시간 스케일링 타입 | linear | linear, exponential, sinusoidal, constant |

### 모델 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--d_model` | 모델 차원 | 128 |
| `--n_layers` | 레이어 수 | 4 |
| `--ssm_size_base` | SSM 상태 크기 | 64 |
| `--blocks` | SSM 블록 수 | 1 |

### 학습 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--epochs` | 학습 에포크 수 | 50 |
| `--bsz` | 배치 크기 | 32 |
| `--lr` | 학습률 | 1e-3 |
| `--weight_decay` | 가중치 감쇠 | 1e-4 |
| `--p_dropout` | 드롭아웃 비율 | 0.1 |

## 📊 시간 스케일링 타입

### 1. Linear (선형)
```
Δ(t) = 1.0 + 0.1 * t / T
```
시간에 따라 선형적으로 증가하는 스케일링

### 2. Exponential (지수)
```
Δ(t) = exp(-0.01 * t)
```
시간에 따라 지수적으로 감소하는 스케일링

### 3. Sinusoidal (사인파)
```
Δ(t) = 1.0 + 0.1 * sin(0.1 * t)
```
사인파 형태로 주기적으로 변화하는 스케일링

### 4. Constant (상수)
```
Δ(t) = 1.0
```
시간에 무관한 상수 스케일링

## 🔬 실험 예시

### 실험 1: 보조 상태 강도 비교
```bash
# 약한 보조 상태
python train_ex.py --enable_auxiliary_state --auxiliary_strength 0.05 --epochs 20

# 중간 보조 상태
python train_ex.py --enable_auxiliary_state --auxiliary_strength 0.1 --epochs 20

# 강한 보조 상태
python train_ex.py --enable_auxiliary_state --auxiliary_strength 0.2 --epochs 20
```

### 실험 2: 시간 스케일링 타입 비교
```bash
# 선형 스케일링
python train_ex.py --enable_auxiliary_state --time_scale_type linear --epochs 20

# 지수 스케일링
python train_ex.py --enable_auxiliary_state --time_scale_type exponential --epochs 20

# 사인파 스케일링
python train_ex.py --enable_auxiliary_state --time_scale_type sinusoidal --epochs 20
```

### 실험 3: 모델 크기 비교
```bash
# 작은 모델
python train_ex.py --d_model 64 --ssm_size_base 32 --epochs 20

# 중간 모델
python train_ex.py --d_model 128 --ssm_size_base 64 --epochs 20

# 큰 모델
python train_ex.py --d_model 256 --ssm_size_base 128 --epochs 20
```

## 📈 결과 해석

### 성능 지표
- **훈련 정확도**: 훈련 데이터에 대한 분류 정확도
- **검증 정확도**: 검증 데이터에 대한 분류 정확도
- **테스트 정확도**: 테스트 데이터에 대한 분류 정확도

### 비교 분석
1. **보조 상태 효과**: `--enable_auxiliary_state` 플래그를 켜고 끈 결과 비교
2. **스케일링 타입 효과**: 다양한 `--time_scale_type` 설정의 성능 비교
3. **강도 효과**: `--auxiliary_strength` 값에 따른 성능 변화

## 🐛 문제 해결

### 일반적인 오류

1. **ImportError**: S5 모듈을 찾을 수 없는 경우
   ```bash
   # s5 디렉토리가 현재 디렉토리에 있는지 확인
   ls s5/
   ```

2. **MemoryError**: GPU 메모리 부족
   ```bash
   # 배치 크기 줄이기
   python train_ex.py --bsz 16
   ```

3. **DataNotFoundError**: ListOps 데이터셋을 찾을 수 없는 경우
   ```bash
   # 데이터 디렉토리 지정
   python train_ex.py --dir_name /path/to/data
   ```

## 📝 참고사항

- ListOps 태스크는 수학적 표현식을 분류하는 태스크입니다
- 보조 상태는 긴 시퀀스에서 정보를 더 잘 보존하는 데 도움이 될 수 있습니다
- 시간 스케일링 타입은 태스크의 특성에 따라 성능이 달라질 수 있습니다
- WandB를 사용하면 실험 결과를 더 쉽게 추적할 수 있습니다

## 🤝 기여

버그 리포트나 기능 제안은 이슈를 통해 제출해주세요. 