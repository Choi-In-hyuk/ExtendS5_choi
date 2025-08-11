#!/usr/bin/env python3
"""
간단한 ListOps 태스크 학습 스크립트
S5SSMWithAuxiliaryState를 사용하여 보조 상태를 포함한 S5 모델을 학습시킵니다.
"""

import argparse
import os
import sys
from functools import partial
from typing import Dict, Any

import jax
import jax.numpy as np
from jax import random
from jax.scipy.linalg import block_diag
import optax
from flax import linen as nn
from flax.training import train_state
import wandb

# S5 모듈 import
from .train_helpers import create_train_state, train_epoch, validate
from .dataloading import Datasets
from .seq_model import BatchClassificationModel
from .ssm_extend import init_S5SSMWithAuxiliaryState
from .ssm_init import make_DPLR_HiPPO


def create_parser():
    """명령행 인자 파서 생성"""
    parser = argparse.ArgumentParser(description='S5SSMWithAuxiliaryState ListOps 학습')
    
    # 기본 모델 파라미터
    parser.add_argument('--d_model', type=int, default=128, help='모델 차원')
    parser.add_argument('--n_layers', type=int, default=4, help='레이어 수')
    parser.add_argument('--ssm_size_base', type=int, default=64, help='SSM 상태 크기')
    parser.add_argument('--blocks', type=int, default=1, help='SSM 블록 수')
    
    # 보조 상태 관련 파라미터
    parser.add_argument('--enable_auxiliary_state', action='store_true', 
                       help='보조 상태 활성화')
    parser.add_argument('--auxiliary_strength', type=float, default=0.1,
                       help='보조 상태 영향 강도')
    parser.add_argument('--time_scale_type', type=str, default='linear',
                       choices=['linear', 'exponential', 'sinusoidal', 'constant'],
                       help='시간 스케일링 타입')
    
    # 학습 파라미터
    parser.add_argument('--epochs', type=int, default=50, help='학습 에포크 수')
    parser.add_argument('--bsz', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='가중치 감쇠')
    parser.add_argument('--p_dropout', type=float, default=0.1, help='드롭아웃 비율')
    
    # SSM 파라미터
    parser.add_argument('--C_init', type=str, default='trunc_standard_normal',
                       help='C 행렬 초기화 방법')
    parser.add_argument('--discretization', type=str, default='zoh',
                       choices=['zoh', 'bilinear'], help='이산화 방법')
    parser.add_argument('--dt_min', type=float, default=0.001, help='최소 시간 스텝')
    parser.add_argument('--dt_max', type=float, default=0.1, help='최대 시간 스텝')
    parser.add_argument('--conj_sym', action='store_true', help='켤레 대칭 사용')
    parser.add_argument('--clip_eigs', action='store_true', help='고유값 클리핑')
    parser.add_argument('--bidirectional', action='store_true', help='양방향 사용')
    
    # 기타
    parser.add_argument('--jax_seed', type=int, default=42, help='JAX 시드')
    parser.add_argument('--dir_name', type=str, default='./data', help='데이터 디렉토리')
    parser.add_argument('--use_wandb', action='store_true', help='WandB 사용')
    parser.add_argument('--wandb_project', type=str, default='s5-auxiliary-state',
                       help='WandB 프로젝트명')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB 엔티티')
    
    return parser


def main():
    """메인 학습 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("S5SSMWithAuxiliaryState ListOps 학습 시작")
    print("=" * 60)
    print(f"보조 상태 활성화: {args.enable_auxiliary_state}")
    print(f"보조 상태 강도: {args.auxiliary_strength}")
    print(f"시간 스케일링 타입: {args.time_scale_type}")
    print("=" * 60)
    
    # WandB 초기화
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"auxiliary_{args.time_scale_type}_{args.auxiliary_strength}"
        )
    else:
        wandb.init(mode='offline')
    
    # 랜덤 시드 설정
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)
    
    # 데이터셋 로드
    print("[*] 데이터셋 로딩 중...")
    create_dataset_fn = Datasets["listops-classification"]
    trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
        create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)
    
    print(f"데이터셋 정보:")
    print(f"  - 시퀀스 길이: {seq_len}")
    print(f"  - 입력 차원: {in_dim}")
    print(f"  - 클래스 수: {n_classes}")
    print(f"  - 훈련 샘플 수: {train_size}")
    
    # SSM 초기화
    print("[*] SSM 초기화 중...")
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    
    # HiPPO 초기화
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
    
    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2
    
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T
    
    # 블록 대각 행렬 생성
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))
    
    print(f"SSM 파라미터:")
    print(f"  - Lambda.shape: {Lambda.shape}")
    print(f"  - V.shape: {V.shape}")
    print(f"  - Vinv.shape: {Vinv.shape}")
    
    # S5SSMWithAuxiliaryState 초기화
    ssm_init_fn = init_S5SSMWithAuxiliaryState(
        H=args.d_model,
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=args.C_init,
        discretization=args.discretization,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        conj_sym=args.conj_sym,
        clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional,
        enable_auxiliary_state=args.enable_auxiliary_state,
        auxiliary_strength=args.auxiliary_strength,
        time_scale_type=args.time_scale_type
    )
    
    # 모델 클래스 생성
    model_cls = partial(
        BatchClassificationModel,
        ssm=ssm_init_fn,
        d_output=n_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        padded=True,  # ListOps는 패딩 사용
        activation='gelu',
        dropout=args.p_dropout,
        mode='pool',
        prenorm=True,
        batchnorm=False,
        bn_momentum=0.9,
    )
    
    # 최적화 설정
    opt_config = {
        'optimizer': 'adamw',
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'lr_schedule': 'cosine',
        'warmup_steps': 1000,
        'max_steps': 100000,
    }
    
    # 훈련 상태 생성
    print("[*] 훈련 상태 초기화 중...")
    state = create_train_state(
        model_cls,
        init_rng,
        padded=True,
        retrieval=False,
        selective_copying=False,
        in_dim=in_dim,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=False,
        opt_config=opt_config,
        ssm_lr=args.lr,
        lr=args.lr,
        dt_global=False
    )
    
    # 훈련 루프
    print("[*] 훈련 시작...")
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # 훈련
        train_loss, train_acc, state = train_epoch(
            state, trainloader, epoch, train_rng, verbose=True
        )
        
        # 검증
        val_loss, val_acc = validate(
            state, valloader, epoch, verbose=True
        )
        
        # 테스트 (마지막 에포크에서만)
        if epoch == args.epochs - 1:
            test_loss, test_acc = validate(
                state, testloader, epoch, verbose=True, prefix="Test"
            )
        
        # 로깅
        log_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        
        if epoch == args.epochs - 1:
            log_dict.update({
                'test_loss': test_loss,
                'test_acc': test_acc,
            })
        
        wandb.log(log_dict)
        
        # 최고 성능 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            print(f"새로운 최고 검증 정확도: {best_val_acc:.4f}")
        
        print(f"훈련 손실: {train_loss:.4f}, 훈련 정확도: {train_acc:.4f}")
        print(f"검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("훈련 완료!")
    print(f"최고 검증 정확도: {best_val_acc:.4f} (에포크 {best_epoch+1})")
    if epoch == args.epochs - 1:
        print(f"최종 테스트 정확도: {test_acc:.4f}")
    print("=" * 60)
    
    wandb.finish()


if __name__ == "__main__":
    main() 