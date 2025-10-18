import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset
from models.hierarchical_cache_attention_model import HCAM  # change model here if needed
import os
from PIL import Image
import numpy as np

# ------------------------
# Config
# ------------------------
MODEL_NAME = "hcam"
CHECKPOINT_PATH = "checkpoints/best_model_epoch_20_val_0.0180.pth"  # update if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT, IMG_WIDTH = 128, 128
IN_CHANNELS = 3
OUT_CHANNELS = 1
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------
# Load model
# ------------------------
model = HCAM(dim=32, inp_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.6f}")

# ------------------------
# Dataset & Loader
# ------------------------
test_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*IN_CHANNELS, std=[0.5]*IN_CHANNELS)
])

test_dataset = CustomDataset(root_dir="Test_Dataset", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------------
# Helper functions
# ------------------------
def denormalize_output(tensor):
    """Convert [0,1] torch tensor to uint8 grayscale"""
    tensor = tensor.squeeze().cpu().clamp(0, 1)  # shape [H,W]
    return (tensor.numpy() * 255).astype(np.uint8)

# ------------------------
# Inference & Saving
# ------------------------
with torch.no_grad():
    for idx, (inputs, target) in enumerate(test_loader):
        inputs, target = inputs.to(DEVICE), target.to(DEVICE)

        # Forward pass
        output = model(inputs)
        output = torch.clamp(output, 0, 1)

        # Convert to image
        pred_img = denormalize_output(output)

        # Resize prediction back to GT size
        if isinstance(target, torch.Tensor):
            gt_size = target.squeeze().cpu().numpy().shape  # (H, W)
        else:
            gt_size = target.shape

        pred_pil = Image.fromarray(pred_img).resize(gt_size[::-1], Image.BILINEAR)

        # Save prediction
        save_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_pred_{idx:04d}.png")
        pred_pil.save(save_path)
        print(f"Saved: {save_path}")
