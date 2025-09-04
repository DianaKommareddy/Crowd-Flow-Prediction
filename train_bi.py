import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from dataset import CrowdFlowDataset, default_transform
from models.BioInspiredModel import BioCrowdFlowModel


# -----------------------------
# Device Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# Dataset & DataLoader
# -----------------------------
train_dataset = CrowdFlowDataset(
    root_dir="Train_Dataset",   # 👈 change path if needed
    transform=default_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,  # adjust to GPU memory
    shuffle=True,
    num_workers=2
)


# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = BioCrowdFlowModel().to(device)

criterion = nn.MSELoss()  # assuming regression on Y (density/flow map)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# -----------------------------
# Training Loop
# -----------------------------
num_epochs = 20  # adjust as needed
save_path = "checkpoints/bioinspired_best.pth"
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        A, E, G, Y = batch
        A, E, G, Y = A.to(device), E.to(device), G.to(device), Y.to(device)

        # Forward
        outputs = model(A, E, G)
        loss = criterion(outputs, Y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save best model
    torch.save(model.state_dict(), save_path)

print("Training complete! Model saved at:", save_path)
