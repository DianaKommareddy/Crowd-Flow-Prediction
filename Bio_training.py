import torch
from torch.utils.data import DataLoader
import os
from dataset import CustomDataset
from models.BioInspiredModel import BioCrowdFlowModel
import torchvision.utils as vutils
import torch.nn.functional as F

# SSIM from pytorch-ssim (install via: pip install pytorch-ssim)
try:
    import pytorch_ssim
    has_ssim = True
except ImportError:
    print("⚠️ pytorch-ssim not installed, SSIM will be skipped. Run: pip install pytorch-ssim")
    has_ssim = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Dataset and Model Setup
# -----------------------------
test_dataset = CustomDataset(root_dir="dataset")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

model = BioCrowdFlowModel(
    dim=32,
    heads=4,
    groups=2,
    num_latents=4,
    decoder_depth=2,
    saliency=True    # <---- ENSURE THIS IS TRUE
)
model = model.to(device)
checkpoint_path = "checkpoints/Bioinspired_best.pt"
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded checkpoint from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
model.eval()

# -----------------------------
# Loss Functions
# -----------------------------
mse_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()
ssim_loss = pytorch_ssim.SSIM() if has_ssim else None
total_mse, total_mae, total_ssim = 0.0, 0.0, 0.0

# -----------------------------
# Evaluation Loop
# -----------------------------
os.makedirs("predictions", exist_ok=True)
os.makedirs("saliency_maps", exist_ok=True)  # <-- save saliency images

with torch.no_grad():
    for i, (A, E, G, Y) in enumerate(test_loader):
        A, E, G, Y = A.to(device), E.to(device), G.to(device), Y.to(device)
        outputs, saliency = model(A, E, G)      # <--- UNPACK BOTH OUTPUTS

        # Metrics
        mse_val = mse_loss(outputs, Y).item()
        mae_val = mae_loss(outputs, Y).item()
        ssim_val = ssim_loss(outputs, Y).item() if has_ssim else 0.0
        total_mse += mse_val
        total_mae += mae_val
        total_ssim += ssim_val

        # Save predictions, ground truth, and saliency
        vutils.save_image(outputs, f"predictions/pred_{i}.png", normalize=True)
        vutils.save_image(Y, f"predictions/gt_{i}.png", normalize=True)
        if saliency is not None:
            # Make sure the saliency output is of shape [B, 1, H, W]
            vutils.save_image(saliency, f"saliency_maps/saliency_{i}.png", normalize=True)

        print(f"[{i+1}/{len(test_loader)}] MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, SSIM: {ssim_val:.4f}")

# -----------------------------
# Final Results
# -----------------------------
n_samples = len(test_loader)
avg_mse = total_mse / n_samples
avg_mae = total_mae / n_samples
avg_ssim = total_ssim / n_samples if has_ssim else None
print("\n✅ Test finished")
print(f"Average MSE : {avg_mse:.4f}")
print(f"Average MAE : {avg_mae:.4f}")
if avg_ssim is not None:
    print(f"Average SSIM: {avg_ssim:.4f}")
