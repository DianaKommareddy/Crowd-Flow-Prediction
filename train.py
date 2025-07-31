import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CrowdFlowDataset
from models.restormer_crowd_flow import RestormerCrowdFlow
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
dataset = CrowdFlowDataset(root_dir='dataset')

# Train/validation split
val_ratio = 0.1
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
batch_size = 8  # Adjust based on your GPU
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = RestormerCrowdFlow().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Directory for saving best model
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float('inf')
epochs = 10  # Start small and adjust

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    print(f"\n🔁 Epoch [{epoch+1}/{epochs}]")

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f"  Step [{step+1}/{len(train_loader)}] Train Loss: {loss.item():.6f}")

    avg_train_loss = train_loss / len(train_loader)
    print(f"📉 Average Train Loss: {avg_train_loss:.6f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"✅ Validation Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'checkpoints/restormer_best.pth')
        print("💾 Best model saved!")

print("\n🎉 Training complete!")
