# Selective Copying Dataset

            This is a synthetic benchmark dataset designed to evaluate the selective memory and copying capability of sequence models such as Mamba, S5, or RNNs.

            ## Task Description

            Each sequence:
            - Length: 4096
            - Contains mostly 0 (noise)
            - Exactly 16 non-zero valid tokens inserted at random positions (IDs 1–15)

            The model must extract these 16 valid tokens in order.

            ## Format

            - `input`: List[int] of length 4096
            - `target`: List[int] of length 16

            ## Usage

            ```python
            from datasets import load_dataset
            ds = load_dataset("namaewa-im/selective-copying-dataset")
            