import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
from models.restormer_crowd_flow import HINT as RestormerCrowdFlow
import piq
import numpy as np

# Use transforms with resize to 128x128 for the test dataset (same as val)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load test dataset
test_dataset = CustomDataset(root_dir='dataset', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Instantiate model
model = RestormerCrowdFlow(dim=32, inp_channels=9, out_channels=1).to(device)

# Load checkpoint for testing
checkpoint_path = 'checkpoints/best_model_epoch_20_val_0.0180.pth'  # Update as needed
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model from {checkpoint_path}")

# Define loss functions
mae_loss_fn = torch.nn.L1Loss()

# Testing loop
test_mae = []
test_ssim = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = torch.clamp(outputs, 0, 1)  # Clamp outputs for valid range
        test_mae.append(mae_loss_fn(outputs, targets).item())
        test_ssim.append(piq.ssim(outputs, targets, data_range=1.0).item())

avg_test_mae = np.mean(test_mae)
avg_test_ssim = np.mean(test_ssim)

print(f"Test MAE: {avg_test_mae:.6f}")
print(f"Test SSIM: {avg_test_ssim:.4f}")
