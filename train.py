import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import CustomDataset
from models.restormer_crowd_flow import HINT as RestormerCrowdFlow
import os
import numpy as np
from torch.optim.lr_scheduler import StepLR
import piq  # pip install piq

# EarlyStopping Utility
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
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
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
            print(f"EarlyStopping: Saved best model (val_loss={val_loss:.6f}) → {self.path}")

# Total Variation Loss
def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return tv_h + tv_w

# Use transforms with resize to 128x128
train_per_image_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resized from 64x64 to 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_per_image_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Same for validation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset initialization with corrected transform
dataset = CustomDataset(
    root_dir='dataset',
    transform=train_per_image_transform
)

val_ratio = 0.1
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply validation transform to validation dataset
val_dataset.dataset.transform = val_per_image_transform

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Instantiate model
model = RestormerCrowdFlow(dim=32, inp_channels=9, out_channels=1).to(device)

print("Training from scratch without loading any pre-trained weights.")

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

mae_loss_fn = nn.L1Loss()
mse_loss_fn = nn.MSELoss()

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
latest_path = os.path.join(checkpoint_dir, 'restormer_latest.pth')

early_stopper = EarlyStopping(patience=10, min_delta=1e-4)
start_epoch = 0
best_val_loss = float('inf')

for epoch in range(start_epoch, epochs):
    print(f"\nEpoch [{epoch + 1}/{epochs}]")
    model.train()
    train_losses = []

    for step, (inputs, targets) in enumerate(train_loader):
        print(f"Inputs tensor size: {inputs.shape}")  # Confirm input size
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        mae = mae_loss_fn(outputs, targets)
        mse = mse_loss_fn(outputs, targets)
        ssim_val = piq.ssim(outputs, targets, data_range=1.0)
        tv = total_variation_loss(outputs)
        loss = mae + mse + (1 - ssim_val) + 0.001 * tv
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        print(f"  [Train] Step {step + 1}/{len(train_loader)} | Loss: {loss.item():.6f}")

    avg_train_loss = np.mean(train_losses)
    print(f"Train Loss — Avg: {avg_train_loss:.6f}")

    model.eval()
    val_mae = []
    val_ssim = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_mae.append(mae_loss_fn(outputs, targets).item())
            val_ssim.append(piq.ssim(outputs, targets, data_range=1.0).item())

    avg_val_mae = np.mean(val_mae)
    avg_val_ssim = np.mean(val_ssim)
    print(f"Val MAE: {avg_val_mae:.6f} | SSIM: {avg_val_ssim:.4f}")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_mae
    }, latest_path)

    if avg_val_mae < best_val_loss:
        best_val_loss = avg_val_mae
        best_model_path = os.path.join(
            checkpoint_dir,
            f"best_model_epoch_{epoch + 1:02d}_val_{avg_val_mae:.4f}.pth"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }, best_model_path)
        print(f"New best model saved: {best_model_path}")

    early_stopper(avg_val_mae, model, epoch + 1, optimizer)
    if early_stopper.early_stop:
        print("Early stopping triggered. Training halted.")
        break

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Learning Rate: {current_lr:.6f}")

print("\nTraining complete!")
