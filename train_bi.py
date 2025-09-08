import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.BioInspiredModel import BioCrowdFlowModel
from dataset import CustomDataset

# ------------------------------
# Hyperparameters & Setup
# ------------------------------
DATASET_DIR = "Train_Dataset"
SAVE_PATH = "checkpoints/bioinspired_best.pt"
BATCH_SIZE = 4               # small batch size for GPU memory
EPOCHS = 200
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    num_workers=4
)

# ------------------------------
# Loss and Optimizer
# ------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------------
# Training Loop
# ------------------------------
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            if BATCH_SIZE < m.num_groups:
                print(f"[Warning] Batch size {BATCH_SIZE} < num_groups {m.num_groups} in GroupNorm → may destabilize training.")

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)
    for A, E, G, label in loop:
        A, E, G, label = A.to(DEVICE), E.to(DEVICE), G.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(A, E, G)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Best model saved at {SAVE_PATH} (loss={best_loss:.4f})")
