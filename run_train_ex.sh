#!/bin/bash

# S5SSMWithAuxiliaryState MNIST 학습 실행 스크립트

# mamba 환경 활성화
conda activate mamba

echo "S5SSMWithAuxiliaryState MNIST 학습 시작"
echo "========================================"

# MNIST 실험
echo "=== MNIST 실험 ==="
echo "1. MNIST 보조 상태 활성화 (선형 스케일링) 학습..."
python3 -m s5.train_ex \
    --dataset mnist-classification \
    --enable_auxiliary_state \
    --auxiliary_strength 0.1 \
    --time_scale_type linear \
    --epochs 20 \
    --bsz 32 \
    --d_model 128 \
    --n_layers 4 \
    --ssm_size_base 64 \
    --lr 1e-3 \
    --jax_seed 42 \
    --dir_name ./data

echo ""
echo "2. MNIST 보조 상태 활성화 (지수 스케일링) 학습..."
python3 -m s5.train_ex \
    --dataset mnist-classification \
    --enable_auxiliary_state \
    --auxiliary_strength 0.1 \
    --time_scale_type exponential \
    --epochs 20 \
    --bsz 32 \
    --d_model 128 \
    --n_layers 4 \
    --ssm_size_base 64 \
    --lr 1e-3 \
    --jax_seed 42 \
    --dir_name ./data

echo ""
echo "3. MNIST 보조 상태 활성화 (사인파 스케일링) 학습..."
python3 -m s5.train_ex \
    --dataset mnist-classification \
    --enable_auxiliary_state \
    --auxiliary_strength 0.1 \
    --time_scale_type sinusoidal \
    --epochs 20 \
    --bsz 32 \
    --d_model 128 \
    --n_layers 4 \
    --ssm_size_base 64 \
    --lr 1e-3 \
    --jax_seed 42 \
    --dir_name ./data

echo ""
echo "모든 MNIST 실험 완료!"
echo "결과를 비교해보세요." 