import torch
from torch.utils.data import DataLoader
import os
from dataset import CustomDataset
from models.BioInspiredModel import BioCrowdFlowModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use the same CustomDataset as in training, with consistent transform
test_dataset = CustomDataset(root_dir="Test Dataset")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
model = BioCrowdFlowModel(
    dim=32,          # smaller dimension used in training
    heads=4,         # fewer heads as in training
    groups=2,        # fewer groups as in training
    num_latents=4,   # fewer latent tokens as in training
    decoder_depth=2  # fewer transformer blocks as in training
)
model = model.to(device)
checkpoint_path = "checkpoints/bioinspired_best.pt"
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
model.eval()

mse_loss = torch.nn.MSELoss()
total_loss = 0.0
with torch.no_grad():
    for i, (A, E, G, Y) in enumerate(test_loader):
        A, E, G, Y = A.to(device), E.to(device), G.to(device), Y.to(device)
        outputs = model(A, E, G)
        loss = mse_loss(outputs, Y)
        total_loss += loss.item()
        # Save predictions (optional)
        if i < 10:
            import torchvision.utils as vutils
            os.makedirs("predictions", exist_ok=True)
            vutils.save_image(outputs, f"predictions/pred_{i}.png", normalize=True)
            vutils.save_image(Y, f"predictions/gt_{i}.png", normalize=True)
avg_loss = total_loss / len(test_loader)
print(f"✅ Test finished | Avg Loss: {avg_loss:.4f}")
