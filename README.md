
### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/namaewa-im/ExtendS5.git
cd ExtendS5

# yaml 파일로 가상환경 생성
conda env create -f s5_environment.yaml
conda activate s5

# 또는 수동으로 환경 생성
conda create -n s5 python=3.9
conda activate s5

```

### 2. 데이터 다운로드

```bash
# LRA 데이터셋 다운로드
./bin/download_lra.sh

# 또는 모든 데이터셋 다운로드
./bin/download_all.sh
```

### 3. 기본 실행

```bash
# cifar10 분류 (보조 상태 비활성화)
python -m s5.train_ex --dataset cifar-classification --epochs 20 --bsz 32

# 보조 상태 활성화 (absorbed 모드, 선형 스케일링링)
python -m s5.train_ex --dataset cifar-classification --enable_auxiliary --aux_mode absorbed --delta_type linear --epochs 20 --bsz 32

# explicit 모드, 지수 스케일링
python -m s5.train_ex --dataset cifar-classification --enable_auxiliary --aux_mode explicit --delta_type linear --epochs 20 --bsz 32


# 사인파 스케일링
python -m s5.train_ex --dataset cifar-classification --enable_auxiliary --aux_mode absorbed --delta_type sinusoidal --epochs 20 --bsz 32
```