import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from dataset import CustomDataset
from models.Bio_Transformer import BioCrowdFlowModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Hyperparameters
# -----------------------------
DATASET_DIR = "dataset"        # Your training data directory
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "Bioinspired_best.pt")
BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 1e-4
# -----------------------------
# Dataset and DataLoader
# -----------------------------
train_dataset = CustomDataset(root_dir=DATASET_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# -----------------------------
# Model
# -----------------------------
model = BioCrowdFlowModel(
    dim=32,
    heads=4,
    groups=2,
    num_latents=4,
    decoder_depth=2,
    saliency=True
).to(device)

# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Optional: Early Stopping Parameters
# -----------------------------
PATIENCE = 15
MIN_DELTA = 1e-5
best_loss = float("inf")
patience_counter = 0

# Create checkpoint directory if not exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# Checkpoint loading & resume setup
# -----------------------------
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', float('inf'))
    print(f"Resuming from epoch {start_epoch} with best loss {best_loss:.6f}")
else:
    print("No checkpoint found, starting from scratch.")

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)
    for A, E, G, label in loop:
        A, E, G, label = A.to(device), E.to(device), G.to(device), label.to(device)
        optimizer.zero_grad()
        outputs, saliency = model(A, E, G)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.6f}")

    # Early stopping logic and checkpoint saving
    if avg_loss < best_loss - MIN_DELTA:
        best_loss = avg_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, CHECKPOINT_PATH)
        print(f"✅ Saved improved checkpoint at epoch {epoch+1} with loss {best_loss:.6f}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("🔴 Early stopping triggered.")
        break

print("✅ Training complete.")
