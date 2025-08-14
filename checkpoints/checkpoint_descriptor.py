#!/usr/bin/env python3
"""
체크포인트 파일 구조 확인 스크립트
사용 예시: python checkpoint_descriptor.py --path checkpoints/imdb-classification/best_model.ckpt
"""

import os
import pickle
import argparse
from pathlib import Path
import numpy as np


def show_simple_structure(file_path: str) -> None:
    """
    체크포인트 파일의 구조를 보여줍니다.
    """
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"📁 파일: {os.path.basename(file_path)}")
    print(f"📊 크기: {file_size / (1024*1024):.2f} MB")
    print("=" * 50)
    
    try:
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print_simple_structure(checkpoint)
        
    except Exception as e:
        print(f"❌ 오류: {e}")


def print_simple_structure(obj, indent: int = 0) -> None:
    """
    간단한 구조를 출력합니다.
    """
    prefix = "  " * indent
    
    if isinstance(obj, dict):
        for i, (key, value) in enumerate(obj.items()):
            is_last = i == len(obj) - 1
            tree_char = "└── " if is_last else "├── "
            
            if isinstance(value, dict):
                print(f"{prefix}{tree_char}{key}")
                print_simple_structure(value, indent + 1)
            elif isinstance(value, (list, tuple)):
                if len(value) > 0 and isinstance(value[0], (dict, list, tuple)):
                    print(f"{prefix}{tree_char}{key}")
                    print_simple_structure(value, indent + 1)
                else:
                    print(f"{prefix}{tree_char}{key}: {get_shape_summary(value)}")
            else:
                print(f"{prefix}{tree_char}{key}: {get_shape_summary(value)}")


def get_shape_summary(value) -> str:
    """
    shape 요약을 반환합니다.
    """
    try:
        # 메타데이터 키들은 값으로 출력
        metadata_keys = {'step', 'epoch', 'val_loss', 'val_acc', 'test_loss', 'test_acc'}
        
        # numpy array나 JAX DeviceArray인 경우
        if hasattr(value, 'shape'):
            if hasattr(value, '__array__'):
                # JAX DeviceArray를 numpy로 변환
                arr = np.array(value)
                if arr.size == 1:  # 스칼라 값인 경우
                    return str(float(arr))
                else:
                    return f"shape{arr.shape}"
            else:
                return f"shape{value.shape}"
        elif hasattr(value, '__array__'):
            # JAX DeviceArray를 numpy로 변환
            arr = np.array(value)
            if arr.size == 1:  # 스칼라 값인 경우
                return str(float(arr))
            else:
                return f"shape{arr.shape}"
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "shape(0,)"
            elif isinstance(value[0], (list, tuple)):
                # 중첩된 리스트의 경우
                return f"shape({len(value)}, {len(value[0]) if value[0] else 0})"
            else:
                return f"shape({len(value)},)"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return f"'{value}'"
        else:
            return str(type(value).__name__)
    except:
        return str(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description="체크포인트 파일 간단 구조 확인")
    parser.add_argument("--path", type=str, help="체크포인트 파일 경로")
    
    args = parser.parse_args()
    
    if not args.path:
        # 기본값: 현재 디렉토리에서 best_model.ckpt 찾기
        current_dir = Path.cwd()
        best_model_path = current_dir / "best_model.ckpt"
        
        if best_model_path.exists():
            args.path = str(best_model_path)
        else:
            print("체크포인트 파일 경로를 지정해주세요.")
            print("사용법: python simple_checkpoint_viewer.py --path /path/to/checkpoint.ckpt")
            return
    
    show_simple_structure(args.path)


if __name__ == "__main__":
    main() 