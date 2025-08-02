import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CrowdFlowDataset
from models.restormer_crowd_flow import RestormerCrowdFlow
import os

# ───────────────────────────────────────
# ⚙️ Device Setup
# ───────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ───────────────────────────────────────
# 📦 Load Dataset
# ───────────────────────────────────────
dataset = CrowdFlowDataset(root_dir='dataset')

val_ratio = 0.1
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# ───────────────────────────────────────
# 🧠 Model, Loss, Optimizer
# ───────────────────────────────────────
model = RestormerCrowdFlow().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ───────────────────────────────────────
# 💾 Resume from Checkpoint (if any)
# ───────────────────────────────────────
checkpoint_path = 'checkpoints/restormer_latest.pth'
start_epoch = 0
best_val_loss = float('inf')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['val_loss']
    print(f"✅ Resumed from checkpoint at epoch {start_epoch} with val loss {best_val_loss:.6f}")

# ───────────────────────────────────────
# 🔁 Training Loop
# ───────────────────────────────────────
epochs = 35

for epoch in range(start_epoch, epochs):
    print(f"\n🔁 Epoch [{epoch + 1}/{epochs}]")
    model.train()
    train_loss = 0.0

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
            print(f"  Step [{step + 1}/{len(train_loader)}] Loss: {loss.item():.6f}")

    avg_train_loss = train_loss / len(train_loader)
    print(f"📉 Avg Train Loss: {avg_train_loss:.6f}")

    # ────────────────────────────────
    # ✅ Validation
    # ────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"✅ Validation Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_path = f'checkpoints/restormer_best_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }, best_path)
        print(f"💾 Best model saved at {best_path}")

    # Save latest model (always)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': best_val_loss
    }, checkpoint_path)

print("\n🎉 Training complete!")
