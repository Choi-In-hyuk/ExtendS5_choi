import json
import os
import numpy as np
from datasets import Dataset, DatasetDict

def generate_selective_copying_dataset(num_samples=1000, seq_len=4096, vocab_size=16, num_valid_tokens=16):
    dataset = []
    for _ in range(num_samples):
        x = [0] * seq_len
        valid_tokens = [int(x) for x in np.random.randint(1, vocab_size, size=num_valid_tokens)]
        insert_positions = [int(x) for x in np.random.choice(seq_len, size=num_valid_tokens, replace=False)]
        insert_positions.sort()
        for pos, val in zip(insert_positions, valid_tokens):
            x[pos] = val
        dataset.append({"input": x, "output": valid_tokens})
    return dataset

def save_jsonl(dataset, path):
    with open(path, "w") as f:
        for example in dataset:
            json.dump(example, f)
            f.write("\n")

def main():
    os.makedirs("project/selective_copying_data", exist_ok=True)

    # 1. Generate train/test/val splits
    train_data = generate_selective_copying_dataset(num_samples=1000)
    test_data = generate_selective_copying_dataset(num_samples=200)
    val_data = generate_selective_copying_dataset(num_samples=200)

    # 2. Save to JSONL
    save_jsonl(train_data, "project/selective_copying_data/train.jsonl")
    save_jsonl(test_data, "project/selective_copying_data/test.jsonl")
    save_jsonl(val_data, "project/selective_copying_data/validation.jsonl")

    # 3. Write README
    write_readme("project/selective_copying_data/README.md")

    # 4. Load with HuggingFace Datasets
    train_dataset = Dataset.from_json("project/selective_copying_data/train.jsonl")
    test_dataset = Dataset.from_json("project/selective_copying_data/test.jsonl")
    validation_dataset = Dataset.from_json("project/selective_copying_data/validation.jsonl")
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})

    # 5. Push to the hub
    dataset.push_to_hub("namaewa-im/selective-copying-dataset")

def write_readme(path="selective_copying_data/README.md"):
    content = """# Selective Copying Dataset

            This is a synthetic benchmark dataset designed to evaluate the selective memory and copying capability of sequence models such as Mamba, S5, or RNNs.

            ## Task Description

            Each sequence:
            - Length: 4096
            - Contains mostly 0 (noise)
            - Exactly 16 non-zero valid tokens inserted at random positions (IDs 1–15)

            The model must extract these 16 valid tokens in order.

            ## Format

            - `input`: List[int] of length 4096
            - `output`: List[int] of length 16

            ## Usage

            ```python
            from datasets import load_dataset
            ds = load_dataset("namaewa-im/selective-copying-dataset")
            ```
            """
    with open(path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    main()
