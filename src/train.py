"""Training script for Azure ML."""

import argparse
import time
import torch
import torch.nn as nn


def test_mode():
    """Run infrastructure test without data - just verify GPU works."""
    print("=" * 60)
    print("INFRASTRUCTURE TEST MODE - No data required")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")

    print("\nRunning simple matrix multiplication test...")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    print(f"Matrix multiplication completed in {elapsed:.4f} seconds")
    print(f"Result shape: {c.shape}")

    print("\nRunning a tiny training loop on random data...")
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        x = torch.randn(32, 10, device=device)
        y = torch.randn(32, 1, device=device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch + 1}/5 - Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("INFRASTRUCTURE TEST PASSED - GPU and PyTorch working correctly")
    print("=" * 60)


def train():
    parser = argparse.ArgumentParser(description="PyTorch training script")
    parser.add_argument("--test-mode", action="store_true", help="Run infrastructure test without data")
    parser.add_argument("--dataset-name", type=str, default=None, help="Name of the dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")

    args = parser.parse_args()

    if args.test_mode:
        test_mode()
        return

    if not args.dataset_name:
        raise ValueError("--dataset-name is required when not in --test-mode")

    print(f"Starting training on dataset: {args.dataset_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: Load dataset from Gold storage
    # TODO: Initialize model
    # TODO: Training loop

    print("Training complete!")

if __name__ == "__main__":
    train()
