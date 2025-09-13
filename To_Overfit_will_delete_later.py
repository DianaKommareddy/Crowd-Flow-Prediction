import torch
from torch.utils.data import DataLoader, Subset
import os
from dataset import CustomDataset
from models.Bio_Transformer import BioCrowdFlowModel
import torchvision.utils as vutils
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load full dataset
train_dataset = CustomDataset(root_dir="dataset")

# Create a subset of first N samples for debugging
N = 10
subset = Subset(train_dataset, list(range(N)))

# DataLoader on subset
single_loader = DataLoader(subset, batch_size=1, shuffle=False)

# Model setup (adjust parameters if needed)
model = BioCrowdFlowModel(
    dim=32,
    heads=4,
    groups=2,
    num_latents=4,
    decoder_depth=2,
    saliency=True
).to(device)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Create output directory
os.makedirs("overfit_debug", exist_ok=True)

# Overfit training loop on subset
for epoch in range(300):
    model.train()
    for A, E, G, Y in single_loader:
        A, E, G, Y = A.to(device), E.to(device), G.to(device), Y.to(device)
        optimizer.zero_grad()
        outputs, _ = model(A, E, G)
        # Resize output and label to fixed size 140x140
        outputs = F.interpolate(outputs, size=(140, 140), mode='bilinear', align_corners=False)
        Y = F.interpolate(Y, size=(140, 140), mode='bilinear', align_corners=False)
        # Clamp to valid range
        outputs = torch.clamp(outputs, 0.0, 1.0)
        Y = torch.clamp(Y, 0.0, 1.0)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.6f}")
        # Save predictions and ground truth for visual inspection
        vutils.save_image(outputs, f"overfit_debug/output_epoch{epoch}.png", normalize=False)
        vutils.save_image(Y, f"overfit_debug/ground_truth.png", normalize=False)

print("Overfitting debug finished. Inspect outputs in overfit_debug/ to check if output matches ground truth.")
