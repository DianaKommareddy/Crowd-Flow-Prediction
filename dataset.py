import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.A_dir = os.path.join(root_dir, "A")
        self.E_dir = os.path.join(root_dir, "E")
        self.G_dir = os.path.join(root_dir, "G")
        self.Y_dir = os.path.join(root_dir, "Y")
        self.image_files = sorted(os.listdir(self.A_dir))
        if transform is None:
            # For grayscale images
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Always open as grayscale for model channel agreement
        A_path = os.path.join(self.A_dir, self.image_files[idx])
        E_path = os.path.join(self.E_dir, self.image_files[idx])
        G_path = os.path.join(self.G_dir, self.image_files[idx])
        Y_path = os.path.join(self.Y_dir, self.image_files[idx])
        # Open as L (grayscale, 1 channel)
        A = Image.open(A_path).convert("L")
        E = Image.open(E_path).convert("L")
        G = Image.open(G_path).convert("L")
        Y = Image.open(Y_path).convert("L")
        A = self.transform(A)
        E = self.transform(E)
        G = self.transform(G)
        Y = self.transform(Y)
        return A, E, G, Y
