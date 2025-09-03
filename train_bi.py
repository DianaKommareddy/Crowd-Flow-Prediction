# train_bi.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CrowdFlowDataset
from models import BioCrowdFlowModel
import os
import numpy as np
import pytorch_ssim  # SSIM Loss


# ───────────────────────────────────────
# EarlyStopping Utility
# ───────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, path='checkpoints/best_model.pth', verbose=True):
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
            print(f"✔️ EarlyStopping: Saved best model (val_loss={val_loss:.6f}) → {self.path}")


# ───────────────────────────────────────
# Total Variation Loss
# ───────────────────────────────────────
def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return tv_h + tv_w


# ───────────────────────────────────────
# Helper: unpack batch into A, E, G, target
# ───────────────────────────────────────
def unpack_batch(batch, device):
    """
    Accepts whatever a dataset returns and converts it into A, E, G, target (all tensors on device).
    Supported formats:
      - (A, E, G, target)
      - (inputs, target) where inputs can be:
          * shape (B, 3, H, W) -> channels correspond to A,E,G (split into single-channel tensors)
          * shape (B, 1, H, W) -> assumed A only; E and G returned as zeros
      - dict with keys 'A','E','G','target' or similar (best-effort)
    """
    # case: dict-like
    if isinstance(batch, dict):
        # try common keys
        A = batch.get('A') or batch.get('agent') or batch.get('a')
        E = batch.get('E') or batch.get('env') or batch.get('e')
        G = batch.get('G') or batch.get('goal') or batch.get('g')
        T = batch.get('target') or batch.get('y') or batch.get('label') or batch.get('gt')
        if A is None:
            raise ValueError("Dataset dict must contain keys like 'A','E','G','target' or similar.")
        A, E, G, T = A.to(device), (E.to(device) if E is not None else None), (G.to(device) if G is not None else None), T.to(device)
        if E is None:
            E = torch.zeros_like(A)
        if G is None:
            G = torch.zeros_like(A)
        return A, E, G, T

    # case: tuple/list
    if isinstance(batch, (tuple, list)):
        if len(batch) == 4:
            A, E, G, T = batch
            return A.to(device), E.to(device), G.to(device), T.to(device)
        elif len(batch) == 2:
            inputs, T = batch
            inputs = inputs.to(device)
            T = T.to(device)
            # inputs could be (B,3,H,W) or (B,1,H,W)
            if inputs.dim() == 4 and inputs.shape[1] == 3:
                # split channels into single-channel tensors
                A = inputs[:, 0:1, :, :]
                E = inputs[:, 1:2, :, :]
                G = inputs[:, 2:3, :, :]
                return A, E, G, T
            elif inputs.dim() == 4 and inputs.shape[1] == 1:
                A = inputs
                E = torch.zeros_like(A)
                G = torch.zeros_like(A)
                return A, E, G, T
            else:
                # fallback: try to split along channel into three equal parts
                c = inputs.shape[1]
                if c % 3 == 0:
                    ch = c // 3
                    A = inputs[:, 0:ch, :, :]
                    E = inputs[:, ch:2*ch, :, :]
                    G = inputs[:, 2*ch:3*ch, :, :]
                    # if channels >1, project to single-channel by taking mean across channel dim
                    if A.shape[1] != 1:
                        A = A.mean(dim=1, keepdim=True)
                        E = E.mean(dim=1, keepdim=True)
                        G = G.mean(dim=1, keepdim=True)
                    return A, E, G, T
                else:
                    # last resort: treat inputs as A and create empty E,G
                    A = inputs
                    E = torch.zeros_like(A)
                    G = torch.zeros_like(A)
                    return A, E, G, T

    raise ValueError("Unsupported batch format returned by dataset.")


# ───────────────────────────────────────
# Device Setup
# ───────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ───────────────────────────────────────
# Load Dataset
# ───────────────────────────────────────
dataset = CrowdFlowDataset(root_dir='Train_Dataset')

val_ratio = 0.1
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)


# ───────────────────────────────────────
# Model, Optimizer, Losses
# ───────────────────────────────────────
model = BioCrowdFlowModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-4)
l1_loss_fn = nn.L1Loss()


# ───────────────────────────────────────
# Checkpoint Setup
# ───────────────────────────────────────
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')

start_epoch = 0
best_val_loss = float('inf')

if os.path.exists(latest_path):
    checkpoint = torch.load(latest_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['val_loss']
    print(f"Resumed from checkpoint at epoch {start_epoch} with val loss {best_val_loss:.6f}")


# ───────────────────────────────────────
# Init EarlyStopping
# ───────────────────────────────────────
early_stopper = EarlyStopping(patience=10, min_delta=1e-4)


# ───────────────────────────────────────
# Training Loop
# ───────────────────────────────────────
epochs = 200
for epoch in range(start_epoch, epochs):
    print(f"\n Epoch [{epoch + 1}/{epochs}]")
    model.train()
    train_losses = []

    for step, batch in enumerate(train_loader):
        # unpack and move to device (robust to different dataset outputs)
        try:
            A, E, G, targets = unpack_batch(batch, device)
        except Exception as e:
            print("Error unpacking batch:", e)
            raise

        optimizer.zero_grad()
        outputs = model(A, E, G)

        l1 = l1_loss_fn(outputs, targets)
        # pytorch_ssim.ssim returns a tensor; ensure scalar
        try:
            ssim_val = pytorch_ssim.ssim(outputs, targets)
            if isinstance(ssim_val, torch.Tensor):
                ssim_val = ssim_val.mean()
        except Exception:
            # fallback: if ssim fails for some shapes, set to zero so only L1 applied
            ssim_val = torch.tensor(0.0, device=device)

        tv = total_variation_loss(outputs)

        # Final Loss: combine pixel, perceptual, and smoothness constraints
        loss = l1 + (1 - ssim_val) + 0.001 * tv

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses) if len(train_losses) > 0 else float('nan')
    print(f"Train Loss — Avg: {avg_train_loss:.6f}")

    # ────────────────────────────────
    # Validation
    # ────────────────────────────────
    model.eval()
    val_l1 = []
    val_ssim = []

    with torch.no_grad():
        for batch in val_loader:
            A, E, G, targets = unpack_batch(batch, device)
            outputs = model(A, E, G)
            l1 = l1_loss_fn(outputs, targets)
            try:
                ssim_score = pytorch_ssim.ssim(outputs, targets)
                if isinstance(ssim_score, torch.Tensor):
                    ssim_score = ssim_score.mean().item()
            except Exception:
                ssim_score = 0.0
            val_l1.append(l1.item())
            val_ssim.append(ssim_score)

    avg_val_l1 = np.mean(val_l1) if len(val_l1) > 0 else float('nan')
    avg_val_ssim = np.mean(val_ssim) if len(val_ssim) > 0 else float('nan')

    print(f"Val L1: {avg_val_l1:.6f} | SSIM: {avg_val_ssim:.4f}")

    # ────────────────────────────────
    # Save latest and best checkpoints
    # ────────────────────────────────
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_l1
    }, latest_path)

    if avg_val_l1 < best_val_loss:
        best_val_loss = avg_val_l1
        best_model_path = os.path.join(
            checkpoint_dir,
            f"best_model_epoch_{epoch+1:02d}_val_{avg_val_l1:.4f}.pth"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }, best_model_path)
        print(f"New best model saved: {best_model_path}")

    # EarlyStopping check
    early_stopper(avg_val_l1, model, epoch + 1, optimizer)
    if early_stopper.early_stop:
        print("Early stopping triggered. Training halted.")
        break

print("\n Training complete!")
