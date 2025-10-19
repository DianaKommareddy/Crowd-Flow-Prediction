import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=128):
        
        """
        Args:
            root_dir (str): Root directory containing folders 'A', 'E', 'G', 'Y'
            transform (torchvision.transforms, optional): Transform for input images
            target_size (int): Size to resize both inputs and target (square)
        """
        self.root_dir = root_dir
        self.A_dir = os.path.join(root_dir, "A")
        self.E_dir = os.path.join(root_dir, "E")
        self.G_dir = os.path.join(root_dir, "G")
        self.Y_dir = os.path.join(root_dir, "Y")
        self.image_files = sorted(os.listdir(self.A_dir))
        self.target_size = target_size

        # Input transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.target_size, self.target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # Target transform (grayscale)
        self.target_transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Build paths
        A_path = os.path.join(self.A_dir, self.image_files[idx])
        E_path = os.path.join(self.E_dir, self.image_files[idx])
        G_path = os.path.join(self.G_dir, self.image_files[idx])
        Y_path = os.path.join(self.Y_dir, self.image_files[idx])

        # Load images
        A = Image.open(A_path).convert("RGB")
        E = Image.open(E_path).convert("RGB")
        G = Image.open(G_path).convert("RGB")
        Y = Image.open(Y_path).convert("L")  # Grayscale target

        # Apply transforms
        A = self.transform(A)
        E = self.transform(E)
        G = self.transform(G)
        Y = self.target_transform(Y)

        # Concatenate input channels: [A, E, G] → 9 channels
        input_tensor = torch.cat([A, E, G], dim=0)

        return input_tensor, Y
