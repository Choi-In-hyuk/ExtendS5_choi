#!/usr/bin/env python3
"""
Checkpoint file structure viewer
Usage: python3 check_checkpoint.py
"""

import os
import pickle
from pathlib import Path
import numpy as np


# ===== Absolute checkpoint path =====
DEFAULT_CKPT_PATH = "/home/choi/ExtendS5/checkpoints/imdb-classification/best_model.ckpt"


def show_simple_structure(file_path: str) -> None:
    """
    Show the structure of the checkpoint file
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"File: {os.path.abspath(file_path)}")
    print(f"Size: {file_size / (1024*1024):.2f} MB")
    print("=" * 50)
    
    try:
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print_simple_structure(checkpoint)
        
    except Exception as e:
        print(f"Error: {e}")


def print_simple_structure(obj, indent: int = 0) -> None:
    """
    Print simplified structure of nested objects
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
    Return shape or value summary
    """
    try:
        if hasattr(value, 'shape'):
            arr = np.array(value)
            if arr.size == 1:
                return str(float(arr))
            return f"shape{arr.shape}"
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "shape(0,)"
            elif isinstance(value[0], (list, tuple)):
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
    ckpt_path = str(Path(DEFAULT_CKPT_PATH).resolve())
    show_simple_structure(ckpt_path)


if __name__ == "__main__":
    main()
