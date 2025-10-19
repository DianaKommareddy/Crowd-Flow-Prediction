import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset
from models.hierarchical_cache_attention_model import HCAM
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ------------------------
# Configuration
# ------------------------
checkpoint_path = "checkpoints/best_model_epoch_48_val_0.0110.pth"  # update if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Load Model
# ------------------------
# ✅ Make sure input channels match your dataset (9 channels)
model = HCAM(dim=32, inp_channels=9, out_channels=1).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.6f}")

# ------------------------
# Dataset & Loader
# ------------------------
# Same normalization as in training
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

test_dataset = CustomDataset(root_dir="Testing", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

# ------------------------
# Helper Functions
# ------------------------
def tensor_to_uint8(tensor):
    """Convert a tensor in [0,1] range to uint8 numpy array"""
    tensor = tensor.squeeze().detach().cpu().clamp(0, 1)
    return (tensor.numpy() * 255).astype(np.uint8)

# ------------------------
# Output Directory
# ------------------------
save_dir = "results_SF"
os.makedirs(save_dir, exist_ok=True)

# ------------------------
# Inference & Evaluation
# ------------------------
total_mae = 0.0
total_ssim = 0.0
count = 0

print("\nStarting inference...")

with torch.no_grad():
    for idx, (inputs, target) in enumerate(test_loader):
        inputs, target = inputs.to(device), target.to(device)

        # Forward pass
        with torch.amp.autocast('cuda'):
            output = model(inputs)

        # Clamp prediction to valid range
        output = torch.clamp(output, 0, 1)

        # Convert to uint8 grayscale
        pred_img = tensor_to_uint8(output)
        gt_img = tensor_to_uint8(target)

        # Match dimensions for metric computation
        gt_size = gt_img.shape  # (H, W)
        pred_pil = Image.fromarray(pred_img).resize(gt_size[::-1], Image.BILINEAR)
        pred_resized = np.array(pred_pil)

        # Compute metrics
        mae = np.mean(np.abs(pred_resized.astype(np.float32) - gt_img.astype(np.float32)))
        ssim_val = ssim(pred_resized, gt_img, data_range=255)

        total_mae += mae
        total_ssim += ssim_val
        count += 1

        # Save output
        save_path = os.path.join(save_dir, f"pred_{idx:04d}.png")
        pred_pil.save(save_path)

        print(f"[{idx+1}/{len(test_loader)}] Saved → {save_path} | MAE={mae:.4f} | SSIM={ssim_val:.4f}")

# ------------------------
# Final Results
# ------------------------
avg_mae = total_mae / count
avg_ssim = total_ssim / count

print("\n==================== RESULTS ====================")
print(f"✅ Average MAE:  {avg_mae:.6f}")
print(f"✅ Average SSIM: {avg_ssim:.6f}")
print("=================================================")
