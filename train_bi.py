import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.BioInspiredModel import BioCrowdFlowModel
from dataset import CustomDataset
from torch.cuda.amp import GradScaler, autocast

# ------------------------------
# Hyperparameters & Setup
# ------------------------------
DATASET_DIR = "dataset"
SAVE_PATH = "checkpoints/bioinspired_best.pt"
BATCH_SIZE = 64   # try doubling batch size for fewer steps if GPU allows
EPOCHS = 200
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True   # allow TensorFloat-32 (faster matmul on Ampere+)
torch.backends.cudnn.allow_tf32 = True

# Early stopping parameters
PATIENCE = 10
MIN_DELTA = 1e-5

# ------------------------------
# Model
# ------------------------------
model = BioCrowdFlowModel(
    dim=32,
    heads=4,
    groups=2,
    num_latents=4,
    decoder_depth=2
).to(DEVICE)

# Channels-last memory format (helps on Ampere+ GPUs)
model = model.to(memory_format=torch.channels_last)

# PyTorch 2.0 compile (speeds up training, esp. on GPU)
if hasattr(torch, "compile"):
    model = torch.compile(model)

# ------------------------------
# Dataset and DataLoader
# ------------------------------
train_dataset = CustomDataset(DATASET_DIR)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,           # reduce workers if CPU is slow
    pin_memory=True,
    prefetch_factor=4,       # prefetch batches
    persistent_workers=True
)

# ------------------------------
# Loss and Optimizer
# ------------------------------
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # AdamW often faster & stabler

# Early stopping
best_loss = float("inf")
patience_counter = 0

# AMP scaler
scaler = GradScaler()

# ------------------------------
# Training Loop
# ------------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)
    for A, E, G, label in loop:
        # Move data async for speed
        A, E, G, label = (
            A.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last),
            E.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last),
            G.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last),
            label.to(DEVICE, non_blocking=True)
        )

        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
        with autocast(dtype=torch.float16):   # force FP16 for extra speed
            outputs = model(A, E, G)
            loss = criterion(outputs, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.6f}")

    # Save best model
    if avg_loss < best_loss - MIN_DELTA:
        best_loss = avg_loss
        patience_counter = 0
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Best model saved at {SAVE_PATH} (loss={best_loss:.6f})")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"🔴 Early stopping after {epoch+1} epochs!")
        print(f"Best loss achieved: {best_loss:.6f}")
        break

print("Training completed!")
