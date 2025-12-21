import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import piq

from dataset import CustomDataset
from models.hierarchical_cache_attention_model import HCAM

# ======================================================
# CONFIG (MUST MATCH TRAINING)
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_ROOT = "Test_Dataset"          # contains A / E / G / Y
CHECKPOINT_PATH = "checkpoints/best_model.pth"
RESULTS_DIR = "results_test"

IMG_SIZE = 128
BATCH_SIZE = 1   # 🚨 REQUIRED for HCAM cache

os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================================================
# TRANSFORMS (MATCH TRAINING)
# ======================================================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ======================================================
# DATASET & LOADER
# ======================================================
test_dataset = CustomDataset(
    root_dir=TEST_ROOT,
    transform=test_transform,
    target_size=IMG_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"✅ Test samples: {len(test_dataset)}")

# ======================================================
# MODEL (EXACTLY SAME AS TRAINING)
# ======================================================
model = HCAM(
    dim=32,                # 🔑 MUST be 32
    inp_channels=9,
    out_channels=1
).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

print("✅ Checkpoint loaded successfully")

# ======================================================
# METRICS
# ======================================================
mae_list = []
ssim_list = []

def tensor_to_uint8(x):
    x = x.squeeze().cpu().numpy()
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x

# ======================================================
# TEST LOOP
# ======================================================
print("\n🚀 Testing started...\n")

with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(inputs)
        outputs = torch.clamp(outputs, 0, 1)

        mae = torch.mean(torch.abs(outputs - targets)).item()
        ssim_val = piq.ssim(outputs, targets, data_range=1.0).item()

        mae_list.append(mae)
        ssim_list.append(ssim_val)

        # Save prediction
        pred_img = tensor_to_uint8(outputs)
        h, w = targets.squeeze().shape
        pred_pil = Image.fromarray(pred_img).resize((w, h), Image.BILINEAR)

        save_path = os.path.join(RESULTS_DIR, f"pred_{idx:04d}.png")
        pred_pil.save(save_path)

        print(
            f"[{idx+1}/{len(test_loader)}] "
            f"Saved {save_path} | MAE={mae:.4f} | SSIM={ssim_val:.4f}"
        )

# ======================================================
# FINAL RESULTS
# ======================================================
print("\n" + "="*50)
print("✅ TEST RESULTS")
print(f"Average MAE  : {np.mean(mae_list):.6f}")
print(f"Average SSIM : {np.mean(ssim_list):.6f}")
print("="*50)
