import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import CustomDataset
from models.hierarchical_cache_attention_model import HCAM  # change to RestormerCrowdFlow if needed
import numpy as np
from torch.optim.lr_scheduler import StepLR
import piq  # pip install piq

# ------------------------
# EarlyStopping Utility
# ------------------------
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, path='checkpoints/best_model_earlystop.pth', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.verbose = verbose

    def __call__(self, val_loss, model, epoch, optimizer):
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, epoch, optimizer):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, self.path)
        if self.verbose:
            print(f"EarlyStopping: Saved best model → {self.path} (val_loss={val_loss:.6f})")

# ------------------------
# Total Variation Loss
# ------------------------
def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return tv_h + tv_w

# ------------------------
# Hyperparameters & Paths
# ------------------------
FIXED_SIZE = 128
BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LATEST_PATH = os.path.join(CHECKPOINT_DIR, 'latest_model.pth')

# ------------------------
# Data Transforms & Loader
# ------------------------
train_transform = transforms.Compose([
    transforms.Resize(FIXED_SIZE),
    transforms.CenterCrop(FIXED_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
val_transform = transforms.Compose([
    transforms.Resize(FIXED_SIZE),
    transforms.CenterCrop(FIXED_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = CustomDataset(root_dir='dataset', transform=train_transform)
val_ratio = 0.1
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ------------------------
# Model, Optimizer, Scheduler, Loss
# ------------------------
model = HCAM(dim=32, inp_channels=3, out_channels=1).to(DEVICE)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
mae_loss_fn = nn.L1Loss()
mse_loss_fn = nn.MSELoss()

# ------------------------
# Early Stopping
# ------------------------
early_stopper = EarlyStopping(patience=10, min_delta=1e-4)

# ------------------------
# Mixed Precision
# ------------------------
scaler = torch.cuda.amp.GradScaler()

best_val_loss = float('inf')

# ------------------------
# Training Loop
# ------------------------
for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    model.train()
    train_losses = []

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            mae = mae_loss_fn(outputs, targets)
            mse = mse_loss_fn(outputs, targets)
            ssim_val = piq.ssim(outputs, targets, data_range=1.0)
            tv_loss = total_variation_loss(outputs)
            density_loss = torch.abs(torch.sum(outputs) - torch.sum(targets)) / targets.numel()
            loss = mae + mse + (1 - ssim_val) + 0.001*tv_loss + 0.05*density_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    print(f"Avg Train Loss: {avg_train_loss:.6f}")

    # ---- Validation ----
    model.eval()
    val_mae, val_ssim, val_density_gt, val_density_pred = [], [], [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                outputs = torch.clamp(outputs, 0, 1)
                val_mae.append(mae_loss_fn(outputs, targets).item())
                val_ssim.append(piq.ssim(outputs, targets, data_range=1.0).item())
                val_density_gt.append(torch.sum(targets).item())
                val_density_pred.append(torch.sum(outputs).item())

    avg_val_mae = np.mean(val_mae)
    avg_val_ssim = np.mean(val_ssim)
    avg_val_density_gt = np.mean(val_density_gt)
    avg_val_density_pred = np.mean(val_density_pred)

    print(f"Val MAE: {avg_val_mae:.6f} | SSIM: {avg_val_ssim:.4f} | GT Density: {avg_val_density_gt:.2f} | Pred Density: {avg_val_density_pred:.2f}")

    # ---- Save Latest & Best ----
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_mae
    }, LATEST_PATH)

    if avg_val_mae < best_val_loss:
        best_val_loss = avg_val_mae
        best_model_path = os.path.join(
            CHECKPOINT_DIR,
            f"best_model_epoch_{epoch+1:02d}_val_{avg_val_mae:.4f}.pth"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }, best_model_path)
        print(f"New best model saved: {best_model_path}")

    # ---- Early Stopping ----
    early_stopper(avg_val_mae, model, epoch+1, optimizer)
    if early_stopper.early_stop:
        print("Early stopping triggered. Training halted.")
        break

    scheduler.step()
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    torch.cuda.empty_cache()

print("\nTraining complete!")
