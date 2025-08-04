import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CrowdFlowDataset
from models.restormer_crowd_flow import RestormerCrowdFlow
import os
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import numpy as np
import csv
from torch.optim.lr_scheduler import StepLR  # ✅ New import for LR scheduler

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
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

# ───────────────────────────────────────
# 🧠 Model, Loss, Optimizer
# ───────────────────────────────────────
model = RestormerCrowdFlow().to(device)
criterion = nn.MSELoss()
mae_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# ───────────────────────────────────────
# 💾 Checkpoint Setup
# ───────────────────────────────────────
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
start_epoch = 0
best_val_loss = float('inf')

latest_path = os.path.join(checkpoint_dir, 'restormer_latest.pth')
if os.path.exists(latest_path):
    checkpoint = torch.load(latest_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['val_loss']
    print(f"✅ Resumed from checkpoint at epoch {start_epoch} with val loss {best_val_loss:.6f}")

# ───────────────────────────────────────
# 📂 Training Log Setup
# ───────────────────────────────────────
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_log.csv')

# Write header if log file doesn't exist
if not os.path.exists(log_file) or start_epoch == 0:
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_MSE_Avg', 'Train_MSE_Min', 'Train_MSE_Max', 'Val_MSE', 'Val_MAE', 'Val_SSIM', 'LR'])

# ───────────────────────────────────────
# 🔁 Training Loop
# ───────────────────────────────────────
epochs = 40

for epoch in range(start_epoch, epochs):
    print(f"\n🔁 Epoch [{epoch + 1}/{epochs}]")
    model.train()
    train_losses = []

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        print(f"  [Train] Step {step+1}/{len(train_loader)} | MSE: {loss.item():.6f}")

    avg_train_loss = np.mean(train_losses)
    min_train_loss = np.min(train_losses)
    max_train_loss = np.max(train_losses)
    print(f"📉 Train Loss — Avg: {avg_train_loss:.6f} | Min: {min_train_loss:.6f} | Max: {max_train_loss:.6f}")

    # ────────────────────────────────
    # ✅ Validation
    # ────────────────────────────────
    model.eval()
    val_losses = []
    val_mae = []
    val_ssim = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            pred = outputs.squeeze().cpu().numpy()
            gt = targets.squeeze().cpu().numpy()

            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
            val_mae.append(mae_fn(outputs, targets).item())
            try:
                val_ssim.append(ssim(pred, gt, data_range=1.0))
            except:
                val_ssim.append(0.0)

    avg_val_loss = np.mean(val_losses)
    avg_val_mae = np.mean(val_mae)
    avg_val_ssim = np.mean(val_ssim)

    print(f"✅ Val MSE: {avg_val_loss:.6f} | MAE: {avg_val_mae:.6f} | SSIM: {avg_val_ssim:.4f}")

    # ────────────────────────────────
    # 💾 Save Checkpoints
    # ────────────────────────────────
    ckpt_name = f"epoch_{epoch+1:02d}_val_{avg_val_loss:.4f}.pth"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss
    }, ckpt_path)
    print(f"💾 Checkpoint saved: {ckpt_name}")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss
    }, latest_path)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }, best_model_path)
        print(f"🏆 New best model saved: {best_model_path}")

    # ────────────────────────────────
    # 📝 Append to Log
    # ────────────────────────────────
    current_lr = scheduler.get_last_lr()[0]
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            f"{avg_train_loss:.6f}",
            f"{min_train_loss:.6f}",
            f"{max_train_loss:.6f}",
            f"{avg_val_loss:.6f}",
            f"{avg_val_mae:.6f}",
            f"{avg_val_ssim:.4f}",
            f"{current_lr:.6f}"
        ])

    # 🔄 Step the scheduler
    scheduler.step()
    print(f"🔧 Learning Rate: {current_lr:.6f}")

print("\n🎉 Training complete!")
