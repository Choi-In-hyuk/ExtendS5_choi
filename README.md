
## 1. 환경 설정

#### 저장소 클론
```bash
git clone https://github.com/namaewa-im/ExtendS5.git
cd ExtendS5
```

#### yaml 파일로 가상환경 생성
```bash
conda env create -f s5_environment.yaml
conda activate s5
```

#### 또는 수동으로 환경 생성
```bash
conda create -n s5 python=3.9
conda activate s5
```

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
python -m s5.train_ex --dataset cifar-classification --epochs 20 --bsz 32
```
#### 보조 상태 활성화 (absorbed 모드, 선형 스케일링링)
```bash
python -m s5.train_ex --dataset cifar-classification --enable_auxiliary --aux_mode absorbed --delta_type linear --epochs 20 --bsz 32
```
#### explicit 모드, 지수 스케일링
```bash
python -m s5.train_ex --dataset cifar-classification --enable_auxiliary --aux_mode explicit --delta_type linear --epochs 20 --bsz 32
```
#### explicit 모드, 사인파 스케일링
```bash
python -m s5.train_ex --dataset cifar-classification --enable_auxiliary --aux_mode explicit --delta_type sinusoidal --epochs 20 --bsz 32
```

## 4. argparse
#### 데이터셋 종류
```bash
--dataset [mnist-classification, cifar-classification, imdb-classification, litsops-classification, pathfinder-classification]
```
#### aux_mode
```bash
--aux_mode [absorbed, explicit]
```
#### Δ(t) type
```bash
--delta_type [linear, exponential, sinusoidal, polynomial, constant]
```

## 추가 자료
### aux mode
<img width="600" height="746" alt="image" src="https://github.com/user-attachments/assets/447cd98b-94d1-42f1-a33c-98cf2cd0b680" />


## Δ(t) type
<img width="733" height="747" alt="image" src="https://github.com/user-attachments/assets/482860a1-3dba-4021-afe2-9f0c5a6be654" />

