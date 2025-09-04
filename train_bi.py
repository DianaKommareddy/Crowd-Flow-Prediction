import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

from dataset import CrowdFlowDataset
from models.BioInspiredModel import BioCrowdFlowModel

# ─────────────────────────────────────────────
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─────────────────────────────────────────────
# Dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # reduced resolution to save memory
    transforms.ToTensor()
])

dataset = CrowdFlowDataset(root_dir="Train_Dataset", transform=transform)

# Split into train/val
val_split = 0.2
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)  # batch=1 to avoid OOM
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

# ─────────────────────────────────────────────
# Model, Loss, Optimizer
model = BioCrowdFlowModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

scaler = torch.cuda.amp.GradScaler()  # mixed precision

# ─────────────────────────────────────────────
# Training loop
num_epochs = 50
best_val_loss = float("inf")
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    # ---- Train ----
    model.train()
    running_loss = 0.0
    for A, E, G, targets in train_loader:
        A, E, G, targets = A.to(device), E.to(device), G.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():   # mixed precision
            outputs = model(A, E, G)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for A, E, G, targets in val_loader:
            A, E, G, targets = A.to(device), E.to(device), G.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(A, E, G)
                loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Scheduler step
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ---- Save best model ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(save_dir, "bioinspired_best.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, save_path)
        print(f"✅ Saved new best model to {save_path}")

print("🎉 Training finished!")
