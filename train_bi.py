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
BATCH_SIZE = 32  # or larger depending on your GPU memory      
EPOCHS = 200
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable cudnn benchmark for fixed size inputs to optimize GPU performance
torch.backends.cudnn.benchmark = True

# Early stopping parameters
PATIENCE = 10               # Number of epochs to wait for improvement
MIN_DELTA = 1e-5           # Minimum change to qualify as improvement

# ------------------------------
# Instantiate smaller model for memory savings
# ------------------------------
model = BioCrowdFlowModel(
    dim=32,          # smaller dimension
    heads=4,         # fewer heads
    groups=2,        # fewer groups
    num_latents=4,   # fewer latent tokens
    decoder_depth=2  # fewer transformer blocks
)
model = model.to(DEVICE)

# ------------------------------
# Dataset and DataLoader
# ------------------------------
train_dataset = CustomDataset(DATASET_DIR)  # uses 64x64 transform internally
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=12,         # adjust per your CPU cores
    pin_memory=True,        # speeds up transfer to GPU
    persistent_workers=True # keeps workers alive to reduce overhead
)

# ------------------------------
# Loss and Optimizer
# ------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------------
# Early stopping setup
# ------------------------------
best_loss = float("inf")
patience_counter = 0

# AMP scaler for mixed precision training
scaler = GradScaler()

# Warn about GroupNorm + batch size issues
for m in model.modules():
    if isinstance(m, nn.GroupNorm):
        if BATCH_SIZE < m.num_groups:
            print(f"[Warning] Batch size {BATCH_SIZE} < num_groups {m.num_groups} in GroupNorm → may destabilize training.")

# ------------------------------
# Training Loop with Early Stopping and AMP
# ------------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)
    for A, E, G, label in loop:
        A, E, G, label = A.to(DEVICE), E.to(DEVICE), G.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            outputs = model(A, E, G)
            loss = criterion(outputs, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.6f}")

    # Check for improvement
    if avg_loss < best_loss - MIN_DELTA:
        best_loss = avg_loss
        patience_counter = 0
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Best model saved at {SAVE_PATH} (loss={best_loss:.6f})")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

    # Early stopping check
    if patience_counter >= PATIENCE:
        print(f"🔴 Early stopping triggered after {epoch+1} epochs!")
        print(f"Best loss achieved: {best_loss:.6f}")
        break

print("Training completed!")
