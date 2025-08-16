#!/usr/bin/env python3
"""
Extended S5 training script with freeze functionality and selective S5 checkpoint loading
- Uses init_ExtendedS5SSM from s5/ssm_extend.py
- Supports parameter freezing by layer and parameter type
- Loads ABCD parameters from S5 checkpoint while keeping Extended parameters randomly initialized
"""

import argparse
import os
import pickle
from functools import partial

from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag
import wandb

from .train_helpers import (
    create_train_state,
    train_epoch,
    validate,
    cosine_annealing,
)
from .dataloading import Datasets
from .seq_model import BatchClassificationModel
from .ssm_init import make_DPLR_HiPPO
from .ssm_extend import init_ExtendedS5SSM


def create_parser():
    parser = argparse.ArgumentParser(description="ExtendedS5SSM training with freeze and S5 checkpoint loading")
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist-classification",
        choices=[
            "mnist-classification",
            "pmnist-classification",
            "cifar-classification",
            "imdb-classification",
            "listops-classification",
            "aan-classification",
            "lra-cifar-classification",
            "pathfinder-classification",
            "pathx-classification",
            "speech35-classification",
            "selective-copying",
        ],
        help="dataset to use",
    )
    parser.add_argument("--dir_name", type=str, default="./data", help="root directory for data")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--jax_seed", type=int, default=42)

    # Model/SSM parameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--p_dropout", type=float, default=0.1)
    parser.add_argument("--ssm_size_base", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=1)

    parser.add_argument("--C_init", type=str, default="lecun_normal", choices=["lecun_normal", "trunc_standard_normal", "complex_normal"]) 
    parser.add_argument("--discretization", type=str, default="bilinear", choices=["zoh", "bilinear"]) 
    parser.add_argument("--dt_min", type=float, default=1e-3)
    parser.add_argument("--dt_max", type=float, default=1e-1)
    parser.add_argument("--conj_sym", action="store_true")
    parser.add_argument("--clip_eigs", action="store_true")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--R", type=int, default=10)

    # SSM type selection
    parser.add_argument(
        "--ssm_type", 
        type=str, 
        default="extend",
        choices=["extend"],
        help="Type of extended SSM to use: 'extend' for auxiliary state extension"
    )

    # Extended options
    parser.add_argument("--delta_mode", type=str, default="learnable", choices=["parameterized", "learnable"])

    # Common delta parameters
    parser.add_argument("--delta_type", type=str, default="learnable", choices=["linear", "exponential", "sinusoidal", "polynomial", "constant", "learnable"]) 
    parser.add_argument("--bound_delta", action="store_true")

    # Freeze parameters
    parser.add_argument("--freeze_layers", type=str, default="", 
                        help="comma-separated layer numbers to freeze (e.g. 1,2,3)")
    parser.add_argument("--freeze_params", type=str, default="", 
                        help="comma-separated param types to freeze (e.g. A,B,C,D)")

    # S5 checkpoint loading
    parser.add_argument("--load_s5_checkpoint", type=str, 
                        default="/home/choi/ExtendS5/checkpoints/imdb-classification/best_model.ckpt", 
                        help="Path to S5 checkpoint file (.ckpt) to load ABCD parameters from")

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="s5-extended")
    parser.add_argument("--wandb_entity", type=str, default=None)

    return parser


def load_s5_abcd_params(checkpoint_path, target_params, n_layers):
    """
    S5 체크포인트에서 ABCD 파라미터만 선택적으로 로드
    """
    if not os.path.exists(checkpoint_path):
        print(f"[!] S5 checkpoint not found: {checkpoint_path}")
        return target_params
    
    try:
        with open(checkpoint_path, 'rb') as f:
            s5_checkpoint = pickle.load(f)
        
        s5_params = s5_checkpoint['params']
        print(f"[*] Loading S5 ABCD parameters from: {checkpoint_path}")
        
        # 로드할 파라미터 매핑
        abcd_param_mapping = {
            'Lambda_re': 'Lambda_re',  # A 행렬 실수부
            'Lambda_im': 'Lambda_im',  # A 행렬 허수부
            'B': 'B',                  # B 행렬
            'C1': 'C1',               # C 행렬 1
            'C2': 'C2',               # C 행렬 2
            'D': 'D',                 # D 행렬
            'log_step': 'log_step'    # 시간 스텝
        }
        
        loaded_count = 0
        for layer_idx in range(n_layers):
            layer_key = f'layers_{layer_idx}'
            
            if layer_key in s5_params['encoder'] and layer_key in target_params['encoder']:
                s5_layer = s5_params['encoder'][layer_key]
                target_layer = target_params['encoder'][layer_key]
                
                # seq 부분의 ABCD 파라미터만 복사
                if 'seq' in s5_layer and 'seq' in target_layer:
                    for param_name, target_name in abcd_param_mapping.items():
                        if param_name in s5_layer['seq'] and target_name in target_layer['seq']:
                            # 파라미터 크기 확인
                            s5_shape = s5_layer['seq'][param_name].shape
                            target_shape = target_layer['seq'][target_name].shape
                            
                            if s5_shape == target_shape:
                                target_params['encoder'][layer_key]['seq'][target_name] = s5_layer['seq'][param_name]
                                loaded_count += 1
                                print(f"    Loaded {layer_key}.seq.{param_name} {s5_shape}")
                            else:
                                print(f"    Shape mismatch for {layer_key}.seq.{param_name}: S5={s5_shape}, Target={target_shape}")
        
        print(f"[*] Successfully loaded {loaded_count} ABCD parameters from S5 checkpoint")
        return target_params
        
    except Exception as e:
        print(f"[!] Failed to load S5 checkpoint: {e}")
        return target_params


def create_ssm_init_fn(args, Lambda, V, Vinv, ssm_size):
    """Create SSM initialization function based on ssm_type."""
    
    if args.ssm_type == "extend":
        # Use original ssm_extend
        ssm_init_fn = init_ExtendedS5SSM(
            H=args.d_model,
            P=ssm_size,
            R=args.R,
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
        )
    else:
        raise ValueError(f"Unknown ssm_type: {args.ssm_type}")
    
    return ssm_init_fn


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Argument validation
    if args.ssm_type == "extend":
        print("Using Extended S5 SSM with Lambda extension")

    # WandB setup
    config = vars(args).copy()
    config["ssm_type_description"] = "Auxiliary state extension (original ssm_extend)"
    
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=config, entity=args.wandb_entity)
    else:
        wandb.init(mode="offline")

    # seed
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # dataset
    create_dataset_fn = Datasets[args.dataset]
    trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
        create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    # dataset-specific padding/model setup
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification", "selective-copying"]:
        padded = True
    else:
        padded = False

    # HiPPO 초기화 및 블록 구성
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    # SSM 초기화 함수 생성 (타입별 분기)
    ssm_init_fn = create_ssm_init_fn(args, Lambda, V, Vinv, ssm_size)

    print(f"[*] Using SSM type: {args.ssm_type}")
    if args.ssm_type == "extend":
        print(f"    - Lambda extension enabled (R={args.R})")
    print(f"    - Delta type: {args.delta_type}")

    # Freeze 설정 출력
    if args.freeze_layers or args.freeze_params:
        print(f"[*] Freeze Configuration:")
        print(f"    - Layers: {args.freeze_layers}")
        print(f"    - Params: {args.freeze_params}")

    # 모델 클래스
    model_cls = partial(
        BatchClassificationModel,
        ssm=ssm_init_fn,
        d_output=n_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        padded=padded,
        activation="gelu",
        dropout=args.p_dropout,
        mode="pool",
        prenorm=True,
        batchnorm=False,
        bn_momentum=0.9,
    )

    # 학습 상태 초기화 (freeze 옵션 포함)
    state = create_train_state(
        model_cls,
        init_rng,
        padded=padded,
        retrieval=False,
        selective_copying=False,
        in_dim=in_dim,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=False,
        opt_config="standard",
        ssm_lr=args.lr,
        lr=args.lr,
        dt_global=False,
        freeze_layers=args.freeze_layers,
        freeze_params=args.freeze_params,
    )

    # S5 체크포인트에서 ABCD 파라미터 로드
    if args.load_s5_checkpoint:
        updated_params = load_s5_abcd_params(
            args.load_s5_checkpoint, 
            state.params, 
            args.n_layers
        )
        state = state.replace(params=updated_params)
        print("[*] Extended S5 model initialized with S5 ABCD parameters + random Extended parameters")

    # 학습률 스케줄 파라미터
    steps_per_epoch = max(1, int(train_size / args.bsz))
    end_step = steps_per_epoch * args.epochs
    lr_params = (cosine_annealing, args.lr, args.lr, 0, end_step, "standard", 1e-6)

    # 학습 루프
    best_val_acc = -1.0
    best_epoch = 0

    print("[*] Training started...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        # 단일 에포크 학습
        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(
            state,
            skey,
            model_cls,
            trainloader,
            seq_len,
            in_dim,
            False,
            lr_params,
        )

        # 검증/테스트
        val_loss, val_acc = validate(state, model_cls, valloader, seq_len, in_dim, False)
        test_loss, test_acc = validate(state, model_cls, testloader, seq_len, in_dim, False)

        # 로그 데이터 구성
        log_data = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "ssm_type": args.ssm_type,
            "lambda_extension": True,
            "R": args.R,
            "loaded_s5_checkpoint": bool(args.load_s5_checkpoint),
        }

        # Freeze 정보 추가
        if args.freeze_layers or args.freeze_params:
            log_data.update({
                "freeze_layers": args.freeze_layers,
                "freeze_params": args.freeze_params,
            })

        wandb.log(log_data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            print(f"new best val acc: {best_val_acc:.4f}")

        print(f"train loss: {float(train_loss):.4f}")
        print(f"val loss: {float(val_loss):.4f}, val acc: {float(val_acc):.4f}")
        print(f"test loss: {float(test_loss):.4f}, test acc: {float(test_acc):.4f}")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")
    print(f"SSM type used: {args.ssm_type}")
    if args.load_s5_checkpoint:
        print(f"S5 checkpoint loaded from: {args.load_s5_checkpoint}")
    
    # Final summary to WandB
    wandb.log({
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_test_acc": float(test_acc),
    })
    
    wandb.finish()


if __name__ == "__main__":
    main()