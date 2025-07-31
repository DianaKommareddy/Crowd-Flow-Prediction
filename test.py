import torch
from torch.utils.data import DataLoader
from dataset import CrowdFlowDataset
from models.restormer_crowd_flow import RestormerCrowdFlow
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load test dataset from 'Test Dataset'
test_dataset = CrowdFlowDataset(root_dir='Test Dataset')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load trained model
model = RestormerCrowdFlow().to(device)
model.load_state_dict(torch.load('checkpoints/restormer_epoch20.pth'))
model.eval()

# Directory for saving predictions
os.makedirs("predictions", exist_ok=True)

# Inference and Visualization
print("\n🔍 Running inference and saving comparison images...")
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)

        pred = outputs.squeeze().cpu().numpy()
        gt = targets.squeeze().cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(gt, cmap='gray')
        axs[0].set_title('Ground Truth')
        axs[0].axis('off')

        axs[1].imshow(pred, cmap='gray')
        axs[1].set_title('Prediction')
        axs[1].axis('off')

        plt.tight_layout()
        plt.savefig(f'predictions/compare_{idx}.png')
        plt.close()
        print(f"✅ Saved: predictions/compare_{idx}.png")

# Metric Calculation
mse_list = []
ssim_list = []

print("\n📊 Calculating MSE and SSIM...")
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)

        pred = outputs.squeeze().cpu().numpy()
        gt = targets.squeeze().cpu().numpy()

        mse = np.mean((pred - gt) ** 2)
        ssim_val = ssim(pred, gt, data_range=1.0)

        mse_list.append(mse)
        ssim_list.append(ssim_val)

print(f"\n📈 Average MSE: {np.mean(mse_list):.6f}")
print(f"📈 Average SSIM: {np.mean(ssim_list):.4f}")
