import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from torch.optim.lr_scheduler import StepLR
import piq  # pip install piq

from dataset import CustomDataset
from models.hierarchical_cache_attention_model import HCAM

# =========================================================
# CONFIGURATION
# =========================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "checkpoint_SF/restormer_latest.pth"
OUTPUT_ROOT = "outputs_resume"
BATCH_SIZE = 2
MAX_EPOCHS = 50          # total epochs to train (including resumed)
INIT_LR = 5e-5
VAL_RATIO = 0.1
IMG_SIZE = (128, 128)
IN_CHANNELS = 9
OUT_CHANNELS = 1

os.makedirs("checkpoint_SF", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================================================
# DATA TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

dataset = CustomDataset(root_dir='Train_Dataset', transform=train_transform)
val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

# =========================================================
# MODEL, OPTIMIZER, SCHEDULER, LOSSES
# =========================================================
model = HCAM(dim=32, inp_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=INIT_LR)
scheduler = StepLR(optimizer, step_size=15, gamma=0.7)
mae_loss_fn = nn.L1Loss()
mse_loss_fn = nn.MSELoss()

# =========================================================
# LOAD CHECKPOINT (Resume Training)
# =========================================================
start_epoch = 0
best_val_loss = float('inf')
best_model_filepath = None

if os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['val_loss']
    scheduler.last_epoch = start_epoch - 1
    print(f"✅ Resumed from checkpoint: {CHECKPOINT_PATH} (epoch {start_epoch}, val_loss={best_val_loss:.6f})")
else:
    print("🚀 No checkpoint found — starting new training from epoch 0.")

# =========================================================
# TRAINING LOOP
# =========================================================
for epoch in range(start_epoch, MAX_EPOCHS):
    print(f"\n🟢 Epoch [{epoch+1}/{MAX_EPOCHS}]")
    model.train()
    train_losses = []

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)

            mae = mae_loss_fn(outputs, targets)
            mse = mse_loss_fn(outputs, targets)
            ssim_val = piq.ssim(outputs, targets, data_range=1.0)

            # Regularization losses
            tv_loss = (
                torch.mean(torch.abs(outputs[:, :, :, :-1] - outputs[:, :, :, 1:])) +
                torch.mean(torch.abs(outputs[:, :, :-1, :] - outputs[:, :, 1:, :]))
            )
            density_loss = torch.abs(torch.sum(outputs) - torch.sum(targets)) / targets.numel()

            loss = mae + mse + (1 - ssim_val) + 0.002 * tv_loss + 0.05 * density_loss

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"  Step [{step+1}/{len(train_loader)}] | Loss: {loss.item():.6f}")

    avg_train_loss = np.mean(train_losses)
    print(f"📉 Train Loss (avg): {avg_train_loss:.6f}")

    # =====================================================
    # VALIDATION
    # =====================================================
    model.eval()
    val_mae, val_ssim = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                outputs = torch.clamp(outputs, 0, 1)
                val_mae.append(mae_loss_fn(outputs, targets).item())
                val_ssim.append(piq.ssim(outputs, targets, data_range=1.0).item())

    avg_val_mae = np.mean(val_mae)
    avg_val_ssim = np.mean(val_ssim)
    print(f"🧾 Val MAE: {avg_val_mae:.6f} | SSIM: {avg_val_ssim:.4f}")

    # =====================================================
    # SAVE CHECKPOINTS
    # =====================================================
    # Always save latest
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_mae
    }, CHECKPOINT_PATH)

    # Save best model if improvement
    if avg_val_mae < best_val_loss:
        best_val_loss = avg_val_mae
        if best_model_filepath and os.path.exists(best_model_filepath):
            os.remove(best_model_filepath)
        best_model_filepath = os.path.join(
            "checkpoints",
            f"best_model_epoch_{epoch+1:02d}_val_{avg_val_mae:.4f}.pth"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }, best_model_filepath)
        print(f"💾 New best model saved: {best_model_filepath}")

    scheduler.step()
    print(f"🔹 Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    torch.cuda.empty_cache()

print("\n✅ Training complete!")
