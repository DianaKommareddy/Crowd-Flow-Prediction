import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
from models.restormer_crowd_flow import HINT as RestormerCrowdFlow
import piq
import torch.nn as nn
from torch.serialization import _open_file_like, _legacy_load

def legacy_torch_load(filename, map_location=None):
    with _open_file_like(filename, "rb") as f:
        return _legacy_load(f, map_location=map_location)

# Image transforms (resize to 128x128, match training)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
test_dataset = CustomDataset(root_dir='dataset', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Instantiate model
model = RestormerCrowdFlow(dim=32, inp_channels=9, out_channels=1).to(device)

# Load checkpoint with legacy loader workaround
checkpoint_path = 'checkpoints/best_model_epoch_20_val_0.0180.pth'
checkpoint = legacy_torch_load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define loss function
mae_loss_fn = nn.L1Loss()

# Testing loop
test_mae = []
test_ssim = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = torch.clamp(outputs, 0, 1)
        test_mae.append(mae_loss_fn(outputs, targets).item())
        test_ssim.append(piq.ssim(outputs, targets, data_range=1.0).item())

print(f"Test MAE: {np.mean(test_mae):.6f}")
print(f"Test SSIM: {np.mean(test_ssim):.4f}")
