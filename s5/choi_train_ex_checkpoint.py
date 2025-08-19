#!/usr/bin/env python3
"""
Extended S5 체크포인트 구조 확인용 스크립트
"""
import pickle
import os
from jax.tree_util import tree_map

def save_checkpoint_for_debug(state, epoch, save_dir="./debug_checkpoints"):
    """디버그용 체크포인트 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'params': state.params,
        'opt_state': state.opt_state,
        'step': state.step,
        'epoch': epoch,
    }
    
    checkpoint_path = os.path.join(save_dir, f"extended_s5_debug_epoch_{epoch}.ckpt")
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"[*] Debug checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def analyze_checkpoint_structure(checkpoint_path):
    """체크포인트 구조 분석"""
    def print_structure(obj, name="", indent=0):
        prefix = "  " * indent
        if hasattr(obj, 'shape'):
            print(f"{prefix}├── {name}: shape{obj.shape}")
        elif isinstance(obj, dict):
            print(f"{prefix}├── {name}")
            for key, value in obj.items():
                print_structure(value, key, indent + 1)
        else:
            print(f"{prefix}├── {name}: {type(obj).__name__}")
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print(f"\n{'='*50}")
        print(f"Extended S5 Checkpoint Structure Analysis")
        print(f"File: {checkpoint_path}")
        print(f"Size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
        print(f"{'='*50}")
        
        if 'params' in checkpoint:
            print_structure(checkpoint['params'], 'params')
        
        print(f"{'='*50}")
        return checkpoint
        
    except Exception as e:
        print(f"[!] Failed to analyze checkpoint: {e}")
        return None

# main 함수에 추가할 코드 조각
# 기존 main() 함수의 학습 루프를 다음과 같이 수정:

# 학습 루프 (수정된 버전)
best_val_acc = -1.0
best_epoch = 0

print("[*] Training started... (Debug mode: 5 epochs)")
for epoch in range(min(args.epochs, 5)):  # 최대 5 epoch만
    print(f"\nEpoch {epoch+1}/5")
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

    print(f"train loss: {float(train_loss):.4f}")
    print(f"val loss: {float(val_loss):.4f}, val acc: {float(val_acc):.4f}")
    print(f"test loss: {float(test_loss):.4f}, test acc: {float(test_acc):.4f}")

    # 5번째 epoch 후 체크포인트 저장 및 분석
    if epoch == 4:  # 5번째 epoch (0-indexed)
        print("\n[*] Saving debug checkpoint for structure analysis...")
        checkpoint_path = save_checkpoint_for_debug(state, epoch)
        extended_checkpoint = analyze_checkpoint_structure(checkpoint_path)
        
        # S5 체크포인트와 비교
        s5_checkpoint_path = "/home/choi/ExtendS5/checkpoints/imdb-classification/best_model.ckpt"
        print(f"\n[*] Comparing with S5 checkpoint...")
        s5_checkpoint = analyze_checkpoint_structure(s5_checkpoint_path)
        
        break  # 구조 분석 후 종료