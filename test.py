import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import CustomDataset
from models.restormer_crowd_flow import HINT as RestormerCrowdFlow
import piq  # ensure piq is installed: pip install piq
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset and DataLoader
test_dataset = CustomDataset(root_dir='dataset', transform=None)  # Use proper transform if needed
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Model initialization
model = RestormerCrowdFlow(dim=32, inp_channels=9, out_channels=1).to(device)

# Path to the checkpoint (update as per your setup)
checkpoint_path = 'checkpoints/best_model_epoch_20_val_0.0180.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model from {checkpoint_path}")

# Loss functions
mae_loss_fn = nn.L1Loss()

# Evaluation
test_mae = []
test_ssim = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = torch.clamp(outputs, 0, 1)  # ensure output is between 0-1
        test_mae.append(mae_loss_fn(outputs, targets).item())
        test_ssim.append(piq.ssim(outputs, targets, data_range=1.0).item())

print(f"Test MAE: {np.mean(test_mae):.6f}")
print(f"Test SSIM: {np.mean(test_ssim):.4f}")
