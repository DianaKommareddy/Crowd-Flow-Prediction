import torch
from torch.utils.data import DataLoader
import os
from dataset import CustomDataset
from models.Bio_Transformer import BioCrowdFlowModel
import torchvision.utils as vutils


# Optional SSIM calculation
try:
    import pytorch_ssim
    has_ssim = True
except ImportError:
    print("⚠️ pytorch-ssim not installed, SSIM will be skipped. Run: pip install pytorch-ssim")
    has_ssim = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Dataset and DataLoader
test_dataset = CustomDataset(root_dir="Test Dataset")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)


# Load Model
model = BioCrowdFlowModel(
    dim=32,
    heads=4,
    groups=2,
    num_latents=4,
    decoder_depth=2
).to(device)


checkpoint_path = "checkpoints/Bioinspired_best.pt"
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded checkpoint from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


model.eval()


# Loss Functions
mse_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()
ssim_loss = pytorch_ssim.SSIM() if has_ssim else None


total_mse, total_mae, total_ssim = 0.0, 0.0, 0.0


# Create directory for saving predictions
os.makedirs("predictions", exist_ok=True)


# Evaluation Loop
with torch.no_grad():
    for i, (A, E, G, Y) in enumerate(test_loader):
        A, E, G, Y = A.to(device), E.to(device), G.to(device), Y.to(device)
        outputs, saliency = model(A, E, G)  # Unpack tuple here


        mse_val = mse_loss(outputs, Y).item()
        mae_val = mae_loss(outputs, Y).item()
        ssim_val = ssim_loss(outputs, Y).item() if has_ssim else 0.0

        # Save only prediction images
        vutils.save_image(outputs, f"predictions/pred_{i}.png", normalize=True)


        print(f"[{i+1}/{len(test_loader)}] MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, SSIM: {ssim_val:.4f}")


# Final Results
n_samples = len(test_loader)
avg_mse = total_mse / n_samples
avg_mae = total_mae / n_samples
avg_ssim = total_ssim / n_samples if has_ssim else None


print("\n✅ Test finished")
print(f"Average MSE : {avg_mse:.4f}")
print(f"Average MAE : {avg_mae:.4f}")
if avg_ssim is not None:
    print(f"Average SSIM: {avg_ssim:.4f}")
