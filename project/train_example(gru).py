import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter

# 1. Load dataset from Hugging Face
def load_hf_dataset(dataset_name="namaewa-im/selective-copying-dataset"):
    dataset = load_dataset(dataset_name)

    if "target" in dataset["train"].column_names:
        dataset = dataset.rename_column("target", "output")

    return dataset["train"], dataset["test"], dataset["validation"]


# 2. Wrap Hugging Face dataset with PyTorch Dataset
class SelectiveCopyingTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]["input"]
        y = self.dataset[idx]["output"]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# 3. Simple GRU-based model (replaceable with Mamba or S5)
class SelectiveCopyingModel(nn.Module):
    def __init__(self, vocab_size=16, d_model=512, input_length=4096, output_length=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.decoder_query = nn.Parameter(torch.randn(output_length, d_model))  # (16, D)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, x):
        x = self.embedding(x)                      # (B, 4096, D)
        memory, _ = self.encoder(x)                # (B, 4096, D)

        # Use decoder queries to attend over memory
        B = x.size(0)
        q = self.decoder_query.unsqueeze(0).expand(B, -1, -1)  # (B, 16, D)
        attn_weights = torch.bmm(q, memory.transpose(1, 2))    # (B, 16, 4096)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        context = torch.bmm(attn_weights, memory)              # (B, 16, D)

        logits = self.classifier(context)  # (B, 16, vocab_size)
        return logits
    
# 4. Training loop
def train(model, train_dataloader, val_dataloader, test_dataloader, device, epochs=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir="runs/selective-copying")

    print(f"Training on {device}")
    print(f"Total epochs: {epochs}")
    print(f"Total batches per epoch: {len(train_dataloader)}")
    print("-" * 50)

    global_step = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # (B, 16, vocab_size)

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = outputs.argmax(dim=-1)  # (B, 16)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            global_step += 1

            if (batch_idx + 1) % 10 == 0:
                acc = 100. * correct / total
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_dataloader)}] "
                      f"Loss: {loss.item():.4f} Acc: {acc:.2f}%")

        train_avg_loss = total_loss / len(train_dataloader)
        train_accuracy = 100. * correct / total
        print(f"\nEpoch [{epoch+1}/{epochs}] Training Summary:")
        print(f"Average Loss: {train_avg_loss:.4f}")
        print(f"Accuracy: {train_accuracy:.2f}%")

        writer.add_scalar("Train/Epoch_Loss", train_avg_loss, epoch)
        writer.add_scalar("Train/Epoch_Accuracy", train_accuracy, epoch)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_outputs = model(inputs)

                loss = criterion(val_outputs.view(-1, val_outputs.size(-1)), targets.view(-1))
                val_loss += loss.item()

                predicted = val_outputs.argmax(dim=-1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.numel()

        val_avg_loss = val_loss / len(val_dataloader)
        val_accuracy = 100. * val_correct / val_total
        print(f"\nValidation Summary:")
        print(f"Average Loss: {val_avg_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.2f}%")

        writer.add_scalar("Val/Epoch_Loss", val_avg_loss, epoch)
        writer.add_scalar("Val/Epoch_Accuracy", val_accuracy, epoch)

        # Testing
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                test_outputs = model(inputs)

                loss = criterion(test_outputs.view(-1, test_outputs.size(-1)), targets.view(-1))
                test_loss += loss.item()

                predicted = test_outputs.argmax(dim=-1)
                test_correct += (predicted == targets).sum().item()
                test_total += targets.numel()

        test_avg_loss = test_loss / len(test_dataloader)
        test_accuracy = 100. * test_correct / test_total
        print(f"\nTest Summary:")
        print(f"Average Loss: {test_avg_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.2f}%")
        print("-" * 50)

        writer.add_scalar("Test/Epoch_Loss", test_avg_loss, epoch)
        writer.add_scalar("Test/Epoch_Accuracy", test_accuracy, epoch)

    writer.close()

# 5. Main entry
def main():
    train_data = load_dataset("json", data_files="/workspace/S52/project/selective_copying_data/train.jsonl")["train"]
    val_data = load_dataset("json", data_files="/workspace/S52/project/selective_copying_data/validation.jsonl")["train"]
    test_data = load_dataset("json", data_files="/workspace/S52/project/selective_copying_data/test.jsonl")["train"]

    train_dataset = SelectiveCopyingTorchDataset(train_data)
    val_dataset = SelectiveCopyingTorchDataset(val_data)
    test_dataset = SelectiveCopyingTorchDataset(test_data)

    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SelectiveCopyingModel()
    device = torch.device("cpu")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params}")

    train(model, dataloader, val_dataloader, test_dataloader, device, epochs=5)

    print(torch.__version__)
    print(torch.cuda.is_available())

if __name__ == "__main__":
    main()
