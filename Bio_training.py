import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from dataset import CustomDataset
from models.Bio_Transformer import BioCrowdFlowModel
from tqdm import tqdm

# Use torchvision SSIM for better compatibility
try:
    from torchvision.metrics import StructuralSimilarityIndexMeasure
    has_ssim = True
    ssim_loss = StructuralSimilarityIndexMeasure().to('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    print("torchvision SSIM not installed, SSIM will be skipped.")
    has_ssim = False
    ssim_loss = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Hyperparameters
# -----------------------------
DATASET_DIR = "dataset"        # Your training data directory
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "Bioinspired_best.pt")
BATCH_SIZE = 4
EPOCHS = 200
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
    dim=64,           # increased from 32
    heads=8,          # increased from 4
    groups=4,         # increased from 2
    num_latents=8,    # increased from 4
    decoder_depth=4,  # increased from 2
    saliency=True
).to(device)

# -----------------------------
# Loss and optimizer
# -----------------------------
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
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
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)
    for A, E, G, label in loop:
        A, E, G, label = A.to(device), E.to(device), G.to(device), label.to(device)

        # Normalize labels if needed (uncomment if labels are 8-bit images)
        # label = label / 255.0

        optimizer.zero_grad()
        outputs, saliency = model(A, E, G)  # Model output

        # Apply sigmoid if your model output layer doesn't have it
        outputs = torch.sigmoid(outputs)

        # Clamp outputs to [0,1]
        outputs = torch.clamp(outputs, 0.0, 1.0)

        # Compute combined loss
        loss = 0.7 * mse_loss(outputs, label) + 0.3 * mae_loss(outputs, label)
        if has_ssim:
            loss += 0.2 * (1 - ssim_loss(outputs, label))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.6f}")

    # Early stopping and checkpoint saving
    if avg_loss < best_loss - MIN_DELTA:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Saved improved checkpoint at epoch {epoch+1} with loss {best_loss:.6f}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training complete.")
