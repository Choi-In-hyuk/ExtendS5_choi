#!/usr/bin/env python3
"""
Checkpoint 기능 테스트 스크립트
"""

import os
import json
import numpy as np
from datetime import datetime

def test_checkpoint_save():
    """체크포인트 저장 기능 테스트"""
    print("=== Checkpoint 기능 테스트 ===")
    
    # 테스트용 체크포인트 디렉토리 생성
    checkpoint_dir = "checkpoints/test_dataset"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 테스트용 체크포인트 정보
    checkpoint_info = {
        "epoch": 10,
        "val_loss": 0.12345,
        "val_acc": 0.9876,
        "test_loss": 0.13456,
        "test_acc": 0.9765,
        "best_epoch": 8,
        "dataset": "test_dataset",
        "timestamp": datetime.now().isoformat(),
        "args": {
            "dataset": "test_dataset",
            "epochs": 100,
            "bsz": 32
        }
    }
    
    # 체크포인트 정보 저장
    checkpoint_path = f"{checkpoint_dir}/checkpoint_info.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_info, f, indent=2)
    
    # 테스트용 모델 가중치 (더미 데이터)
    import pickle
    dummy_checkpoint_data = {
        "params": {
            "layer1": np.random.randn(10, 10),
            "layer2": np.random.randn(10, 5)
        },
        "opt_state": {},
        "step": 100,
        "epoch": 10,
        "val_loss": 0.12345,
        "val_acc": 0.9876
    }
    
    # 모델 가중치 저장
    model_path = f"{checkpoint_dir}/model_epoch_010.ckpt"
    with open(model_path, 'wb') as f:
        pickle.dump(dummy_checkpoint_data, f)
    
    # 최고 성능 모델 저장
    best_model_path = f"{checkpoint_dir}/best_model.ckpt"
    with open(best_model_path, 'wb') as f:
        pickle.dump(dummy_checkpoint_data, f)
    
    print(f"✅ 체크포인트 정보 저장: {checkpoint_path}")
    print(f"✅ 모델 가중치 저장: {model_path}")
    print(f"✅ 최고 성능 모델 저장: {best_model_path}")
    
    # 저장된 파일 확인
    print("\n=== 저장된 파일 확인 ===")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            loaded_info = json.load(f)
        print(f"✅ 체크포인트 정보 로드 성공")
        print(f"   - Epoch: {loaded_info['epoch']}")
        print(f"   - Val Loss: {loaded_info['val_loss']:.5f}")
        print(f"   - Val Acc: {loaded_info['val_acc']:.4f}")
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"✅ 모델 가중치 로드 성공")
        print(f"   - Epoch: {loaded_model.get('epoch', 'N/A')}")
        print(f"   - Val Loss: {loaded_model.get('val_loss', 'N/A')}")
        print(f"   - Val Acc: {loaded_model.get('val_acc', 'N/A')}")
    
    print("\n=== 체크포인트 기능 테스트 완료 ===")

if __name__ == "__main__":
    test_checkpoint_save() 