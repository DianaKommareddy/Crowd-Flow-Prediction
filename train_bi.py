# train_bi.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import CrowdFlowDataset
from models import BioCrowdFlowModel
import numpy as np
import pytorch_ssim  # SSIM Loss

# ───────────────────────────────────────
# Config
# ───────────────────────────────────────
IMG_SIZE = 32
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 200
HEADS = 4
GROUPS = 2
USE_AMP = True
PIN_MEMORY = True
PRINT_BATCH_DEBUG_ONCE = True  # prints shapes of the first parsed batch

# Optional: reduce CUDA fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# ───────────────────────────────────────
# EarlyStopping
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
            print(f"✔️ Saved best model (val_loss={val_loss:.6f}) → {self.path}")


# ───────────────────────────────────────
# Total Variation Loss
# ───────────────────────────────────────
def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return tv_h + tv_w


# ───────────────────────────────────────
# Helper utils
# ───────────────────────────────────────
def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(t, torch.Tensor) for t in x):
        # try stack per-sample list into (B, ...)
        try:
            return torch.stack(x, dim=0).to(device)
        except Exception:
            # fall back: return first item moved (better than crashing)
            return x[0].to(device)
    return x  # non-tensor; let caller handle

def _ensure_nchw(x):
    """Make sure tensor is (B, C, H, W). Adds channel dim if needed."""
    if not isinstance(x, torch.Tensor):
        return x
    if x.dim() == 2:
        # (H, W) -> (1, 1, H, W)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        # could be (C,H,W) or (B,H,W). Heuristic: if first dim is small, treat as C
        if x.shape[0] in (1, 3):  # (C,H,W)
            x = x.unsqueeze(0)
        else:  # (B,H,W)
            x = x.unsqueeze(1)
    elif x.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}; expected 2-4 dims")
    return x

def _split_inputs_tensor(inputs):
    """
    Split a (B,C,H,W) inputs into (A,E,G) single-channel tensors.
    If C==3 -> channels 0,1,2.
    If C==1 -> E,G are zeros.
    If C>3 and divisible by 3 -> average each third to 1 channel.
    Otherwise -> treat whole as A and zeros for E,G.
    """
    B, C, H, W = inputs.shape
    if C == 3:
        A = inputs[:, 0:1]
        E = inputs[:, 1:2]
        G = inputs[:, 2:3]
    elif C == 1:
        A = inputs
        Z = torch.zeros_like(A)
        E, G = Z, Z
    elif C % 3 == 0:
        ch = C // 3
        A = inputs[:, 0:ch].mean(dim=1, keepdim=True)
        E = inputs[:, ch:2*ch].mean(dim=1, keepdim=True)
        G = inputs[:, 2*ch:3*ch].mean(dim=1, keepdim=True)
    else:
        A = inputs.mean(dim=1, keepdim=True)
        Z = torch.zeros_like(A)
        E, G = Z, Z
    return A, E, G


# ───────────────────────────────────────
# Helper: unpack batch (robust)
# ───────────────────────────────────────
_first_debug_print_done = False
def unpack_batch(batch, device):
    global _first_debug_print_done

    # Case 1: dict with keys
    if isinstance(batch, dict):
        A = batch.get('A') or batch.get('agent') or batch.get('a')
        E = batch.get('E') or batch.get('env') or batch.get('e')
        G = batch.get('G') or batch.get('goal') or batch.get('g')
        T = batch.get('target') or batch.get('y') or batch.get('label') or batch.get('gt')
        if A is None:
            raise ValueError("Dataset dict must contain keys like 'A','E','G','target'")
        A, E, G, T = _to_device(A, device), _to_device(E, device), _to_device(G, device), _to_device(T, device)
        A, E, G = _ensure_nchw(A), _ensure_nchw(E), _ensure_nchw(G)
        T = _ensure_nchw(T)
        if E is None: E = torch.zeros_like(A)
        if G is None: G = torch.zeros_like(A)

    # Case 2: tuple/list
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 4:
            A, E, G, T = batch
            A, E, G, T = _to_device(A, device), _to_device(E, device), _to_device(G, device), _to_device(T, device)
            A, E, G, T = _ensure_nchw(A), _ensure_nchw(E), _ensure_nchw(G), _ensure_nchw(T)

        elif len(batch) == 2:
            inputs, T = batch
            # if inputs is a dict with A/E/G
            if isinstance(inputs, dict):
                A = inputs.get('A') or inputs.get('agent') or inputs.get('a')
                E = inputs.get('E') or inputs.get('env') or inputs.get('e')
                G = inputs.get('G') or inputs.get('goal') or inputs.get('g')
                A, E, G = _to_device(A, device), _to_device(E, device), _to_device(G, device)
                A, E, G = _ensure_nchw(A), _ensure_nchw(E), _ensure_nchw(G)
                # fill missing E/G
                if E is None: E = torch.zeros_like(A)
                if G is None: G = torch.zeros_like(A)
            else:
                # inputs could be tensor OR list/tuple of tensors
                inputs = _to_device(inputs, device)
                if isinstance(inputs, (list, tuple)):
                    # per-sample structures: try to collate to tensor
                    if len(inputs) > 0 and isinstance(inputs[0], dict):
                        # list of dicts (unlikely because DataLoader usually collates),
                        # but handle gracefully: stack A/E/G from each item
                        A_list, E_list, G_list = [], [], []
                        for d in inputs:
                            A_list.append(_ensure_nchw(_to_device(d.get('A') or d.get('agent') or d.get('a'), device)))
                            e = d.get('E') or d.get('env') or d.get('e')
                            g = d.get('G') or d.get('goal') or d.get('g')
                            E_list.append(_ensure_nchw(_to_device(e, device)) if e is not None else None)
                            G_list.append(_ensure_nchw(_to_device(g, device)) if g is not None else None)
                        A = torch.cat(A_list, dim=0)
                        if all(t is not None for t in E_list):
                            E = torch.cat(E_list, dim=0)
                        else:
                            E = torch.zeros_like(A)
                        if all(t is not None for t in G_list):
                            G = torch.cat(G_list, dim=0)
                        else:
                            G = torch.zeros_like(A)
                    else:
                        # list/tuple of tensors -> stack
                        inputs = _to_device(inputs, device)
                        inputs = torch.stack(inputs, dim=0) if all(isinstance(t, torch.Tensor) for t in inputs) else inputs
                        if not isinstance(inputs, torch.Tensor):
                            raise ValueError("Could not collate inputs list/tuple into a tensor")
                        inputs = _ensure_nchw(inputs)
                        A, E, G = _split_inputs_tensor(inputs)
                else:
                    # inputs is a Tensor
                    inputs = _ensure_nchw(inputs)
                    A, E, G = _split_inputs_tensor(inputs)

            # targets handling
            T = _to_device(T, device)
            if isinstance(T, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in T):
                try:
                    T = torch.stack(T, dim=0)
                except Exception:
                    T = T[0]
            T = _ensure_nchw(T)

        else:
            raise ValueError(f"Unsupported list/tuple length: {len(batch)}")

    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    # One-time debug
    if PRINT_BATCH_DEBUG_ONCE and not _first_debug_print_done:
        _first_debug_print_done = True
        def shape(t): return None if t is None else tuple(t.shape)
        print(f"[Batch debug] A{shape(A)} E{shape(E)} G{shape(G)} T{shape(T)}")

    return A, E, G, T


# ───────────────────────────────────────
# Device
# ───────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    torch.cuda.empty_cache()

# ───────────────────────────────────────
# Dataset & Loaders
# ───────────────────────────────────────
dataset = CrowdFlowDataset(root_dir='Train_Dataset')
val_size = int(len(dataset) * 0.1)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(PIN_MEMORY and device.type=='cuda')
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=(PIN_MEMORY and device.type=='cuda')
)

# ───────────────────────────────────────
# Model, Optimizer, Loss
# ───────────────────────────────────────
model = BioCrowdFlowModel(dim=64, heads=HEADS, groups=GROUPS, num_latents=8, decoder_depth=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
l1_loss_fn = nn.L1Loss()

# ───────────────────────────────────────
# Checkpoints
# ───────────────────────────────────────
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')

start_epoch, best_val_loss = 0, float('inf')
if os.path.exists(latest_path):
    ckpt = torch.load(latest_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    best_val_loss = ckpt['val_loss']
    print(f"Resumed from epoch {start_epoch} with val loss {best_val_loss:.6f}")

# ───────────────────────────────────────
# EarlyStopping & AMP
# ───────────────────────────────────────
early_stopper = EarlyStopping(patience=10, min_delta=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type=='cuda'))

# ───────────────────────────────────────
# Training Loop
# ───────────────────────────────────────
for epoch in range(start_epoch, EPOCHS):
    print(f"\n Epoch [{epoch + 1}/{EPOCHS}]")
    model.train()
    train_losses = []

    for step, batch in enumerate(train_loader):
        A, E, G, targets = unpack_batch(batch, device)

        # Ensure consistent resolution
        if (A.shape[-2], A.shape[-1]) != (IMG_SIZE, IMG_SIZE):
            A = F.interpolate(A, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            E = F.interpolate(E, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            G = F.interpolate(G, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type=='cuda')):
            outputs = model(A, E, G)

            # Match target size to outputs & ensure NCHW
            targets = _ensure_nchw(targets)
            if targets.shape[2:] != outputs.shape[2:]:
                targets = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)

            l1 = l1_loss_fn(outputs, targets)
            try:
                ssim_val = pytorch_ssim.ssim(outputs, targets)
                ssim_val = ssim_val.mean() if isinstance(ssim_val, torch.Tensor) else torch.tensor(float(ssim_val), device=device)
            except Exception:
                ssim_val = torch.tensor(0.0, device=device)
            tv = total_variation_loss(outputs)
            loss = l1 + (1 - ssim_val) + 0.001 * tv

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    print(f"Train Loss — Avg: {avg_train_loss:.6f}")

    # ────────────────────────────────
    # Validation
    # ────────────────────────────────
    model.eval()
    val_l1, val_ssim = [], []
    with torch.no_grad():
        for batch in val_loader:
            A, E, G, targets = unpack_batch(batch, device)
            if (A.shape[-2], A.shape[-1]) != (IMG_SIZE, IMG_SIZE):
                A = F.interpolate(A, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
                E = F.interpolate(E, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
                G = F.interpolate(G, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

            outputs = model(A, E, G)

            targets = _ensure_nchw(targets)
            if targets.shape[2:] != outputs.shape[2:]:
                targets = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)

            l1 = l1_loss_fn(outputs, targets)
            try:
                ssim_score = pytorch_ssim.ssim(outputs, targets)
                ssim_score = ssim_score.mean().item() if isinstance(ssim_score, torch.Tensor) else float(ssim_score)
            except Exception:
                ssim_score = 0.0

            val_l1.append(l1.item())
            val_ssim.append(ssim_score)

    avg_val_l1, avg_val_ssim = np.mean(val_l1), np.mean(val_ssim)
    print(f"Val L1: {avg_val_l1:.6f} | SSIM: {avg_val_ssim:.4f}")

    # ────────────────────────────────
    # Save checkpoints
    # ────────────────────────────────
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_l1
    }, latest_path)

    if avg_val_l1 < best_val_loss:
        best_val_loss = avg_val_l1
        best_model_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1:02d}_val_{avg_val_l1:.4f}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }, best_model_path)
        print(f"New best model saved: {best_model_path}")

    early_stopper(avg_val_l1, model, epoch + 1, optimizer)
    if early_stopper.early_stop:
        print("Early stopping triggered. Training halted.")
        break

print("\n Training complete!")
