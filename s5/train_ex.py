#!/usr/bin/env python3
"""
Extended S5 실험용 간단 학습 스크립트
- s5/ssm_extend.py 의 init_ExtendedS5SSM 사용
"""

import argparse
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
    parser = argparse.ArgumentParser(description="ExtendedS5SSM training")
    # dataset
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

    # 학습 파라미터
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--jax_seed", type=int, default=42)

    # model/SSM parameters
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
        help="Type of extended SSM to use: 'extend' for auxiliary state extension, 'edf' for E*Delta(t)*F modification, 'none' for no extension"
    )

    # Extended options (for ssm_extend)
    parser.add_argument("--delta_mode", type=str, default="learnable", choices=["parameterized", "learnable"])

    # EDF options (for ssm_edf)
    parser.add_argument("--enable_edf", action="store_true", help="Enable EDF modules (E*Delta(t)*F)")
    parser.add_argument("--edf_std", type=float, default=0.1, help="Standard deviation for EDF matrix initialization")

    # Common delta parameters (used by both ssm types)
    parser.add_argument("--delta_type", type=str, default="learnable", choices=["linear", "exponential", "sinusoidal", "polynomial", "constant", "learnable"]) 
    parser.add_argument("--bound_delta", action="store_true")

    # logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="s5-extended")
    parser.add_argument("--wandb_entity", type=str, default=None)

    return parser


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
    elif args.ssm_type == "none":
        ssm_init_fn = init_S5SSM(
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
    config["ssm_type_description"] = {
        "extend": "Auxiliary state extension (original ssm_extend)",
        "none": "No extension"
    }[args.ssm_type]
    
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
        print(f"    - Lambda extension enabled")
    print(f"    - Delta type: {args.delta_type}")

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

    # 학습 상태 초기화
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
    )

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
        }

        # SSM 타입별 추가 로깅
        if args.ssm_type == "extend":
            log_data.update({
                "lambda_extension": True,
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
    
    # Final summary to WandB
    wandb.log({
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_test_acc": float(test_acc),
    })
    
    wandb.finish()


if __name__ == "__main__":
    main()