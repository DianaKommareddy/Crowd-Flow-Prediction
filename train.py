import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import piq

from dataset import CustomDataset
from models.restormer_crowd_flow import HINT as RestormerCrowdFlow

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model_epoch_20_val_0.0136.pth"   # <-- change to your file
OUTPUT_ROOT = "outputs"
BATCH_SIZE = 1
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ----------------------------
# Model
# ----------------------------
model = RestormerCrowdFlow(dim=32, inp_channels=9, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Load checkpoint safely
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.6f}")
model.eval()


# ----------------------------
# Helper function for testing
# ----------------------------
def run_test(dataset_root, resize_shape=(128, 128), tag="Testing"):
    print(f"\n=== Evaluating {tag} ({resize_shape[0]}x{resize_shape[1]}) ===")

    # Define transforms dynamically (same as training resize)
    test_transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    target_transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor()
    ])

    dataset = CustomDataset(root_dir=dataset_root,
                            transform=test_transform,
                            target_transform=target_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    mae_list, ssim_list = [], []
    output_folder = os.path.join(OUTPUT_ROOT, tag)
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)

            # Align sizes if mismatch
            if outputs.shape != targets.shape:
                outputs = F.interpolate(outputs,
                                        size=targets.shape[-2:],
                                        mode='bilinear',
                                        align_corners=False)

            # Rescale [0,1]
            outputs = torch.clamp(outputs, 0, 1)
            targets = torch.clamp(targets, 0, 1)

            # Save output image
            out_img = outputs[0].cpu().numpy()
            if out_img.shape[0] == 1:  # grayscale
                out_img = out_img.squeeze(0) * 255.0
                Image.fromarray(out_img.astype(np.uint8)).save(
                    os.path.join(output_folder, f"{idx}.png"))
            else:  # RGB
                out_img = out_img.transpose(1, 2, 0) * 255.0
                Image.fromarray(out_img.astype(np.uint8)).save(
                    os.path.join(output_folder, f"{idx}.png"))

            # Metrics
            mae_list.append(torch.mean(torch.abs(outputs - targets)).item())
            ssim_list.append(piq.ssim(outputs, targets, data_range=1.0).item())

    print(f"{tag} -> Average MAE: {np.mean(mae_list):.6f}, Average SSIM: {np.mean(ssim_list):.6f}")


# ----------------------------
# Run on ONE dataset
# ----------------------------
# Example: only test on testing140
#run_test("testing140", resize_shape=(128, 128), tag="Test140")

# If you want to test the other one, comment above and use:
run_test("Testing", resize_shape=(128, 128), tag="Test112")
