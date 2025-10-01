import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset
from models.restormer_crowd_flow import HINT as RestormerCrowdFlow
import os
from PIL import Image

# ------------------------
# Config
# ------------------------
checkpoint_path = "checkpoints/best_model_epoch_20_val_0.0180.pth"  # update if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Load model
# ------------------------
model = RestormerCrowdFlow(dim=32, inp_channels=9, out_channels=1).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.6f}")

# ------------------------
# Dataset & Loader
# ------------------------
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# NOTE: CustomDataset should also return original image size for resizing predictions
test_dataset = CustomDataset(root_dir="Test Dataset", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------------
# Helper functions
# ------------------------
def denormalize_output(tensor):
    """Convert [0,1] torch tensor to uint8 grayscale"""
    tensor = tensor.squeeze().cpu().clamp(0, 1)  # shape [H,W]
    return (tensor.numpy() * 255).astype("uint8")

# Results folder
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# ------------------------
# Inference & Saving
# ------------------------
with torch.no_grad():
    for idx, (inputs, target) in enumerate(test_loader):
        inputs, target = inputs.to(device), target.to(device)

        # Forward pass
        output = model(inputs)
        output = torch.clamp(output, 0, 1)

        # Convert to image
        pred_img = denormalize_output(output)

        # Resize prediction back to GT size
        gt_size = target.squeeze().cpu().numpy().shape  # (H, W)
        pred_pil = Image.fromarray(pred_img).resize(gt_size[::-1], Image.BILINEAR)

        # Save prediction
        save_path = os.path.join(save_dir, f"pred_{idx:04d}.png")
        pred_pil.save(save_path)
        print(f"Saved: {save_path}")
