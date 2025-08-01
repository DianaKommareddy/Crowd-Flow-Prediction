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

# Split into training and validation sets
val_ratio = 0.1
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
batch_size = 4  # ⬅️ Lowered to reduce GPU memory (adjust if needed)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# ───────────────────────────────────────
# 🧠 Model, Loss, Optimizer
# ───────────────────────────────────────
model = RestormerCrowdFlow().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ───────────────────────────────────────
# 💾 Training Setup
# ───────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float('inf')
epochs = 40  
# ───────────────────────────────────────
# 🔁 Training Loop
# ───────────────────────────────────────
for epoch in range(epochs):
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

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'checkpoints/restormer_best.pth')
        print("💾 Best model saved!")

print("\n🎉 Training complete!")
