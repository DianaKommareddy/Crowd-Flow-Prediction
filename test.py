import torch
from torch.utils.data import DataLoader
from dataset import CrowdFlowDataset
from models.restormer_crowd_flow import SharpRestormer as RestormerCrowdFlow
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ───────────────────────────────────────
# Device Setup
# ───────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ───────────────────────────────────────
# Load Test Dataset
# ───────────────────────────────────────
test_dataset = CrowdFlowDataset(root_dir='Test Dataset')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ───────────────────────────────────────
# Load Model and Checkpoint
# ───────────────────────────────────────
model = RestormerCrowdFlow().to(device)
checkpoint = torch.load('checkpoints/Bioinspired_best.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ───────────────────────────────────────
# Create Output Directory
# ───────────────────────────────────────
pred_dir = "predictions"
os.makedirs(pred_dir, exist_ok=True)

# ───────────────────────────────────────
# Inference, Save Predictions, and Calculate Metrics
# ───────────────────────────────────────
mse_list = []
ssim_list = []
mae_list = []

print("\nRunning inference and saving predictions...")
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Convert prediction and GT to numpy
        pred = outputs.squeeze().cpu().numpy()
        gt = targets.squeeze().cpu().numpy()

        # Clip values to [0,1]
        pred = np.clip(pred, 0, 1)
        gt = np.clip(gt, 0, 1)

        # Resize to 112×112 with sharp edges
        pred_resized = cv2.resize(pred, (112, 112), interpolation=cv2.INTER_NEAREST)
        gt_resized = cv2.resize(gt, (112, 112), interpolation=cv2.INTER_NEAREST)

        # Save prediction image
        save_path = os.path.join(pred_dir, f"pred_{idx}.png")
        cv2.imwrite(save_path, (pred_resized * 255).astype(np.uint8))
        print(f"Saved: {save_path}")

        # Metrics
        mse = np.mean((pred_resized - gt_resized) ** 2)
        mae = np.mean(np.abs(pred_resized - gt_resized))
        ssim_val = ssim(pred_resized, gt_resized, data_range=1.0)

        mse_list.append(mse)
        mae_list.append(mae)
        ssim_list.append(ssim_val)

#------------------------------------------------------
# Print final metrics
#------------------------------------------------------
print(f"\nAverage MSE: {np.mean(mse_list):.6f}")
print(f"Average MAE: {np.mean(mae_list):.6f}")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")
