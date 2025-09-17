import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.A_dir = os.path.join(root_dir, "A")
        self.E_dir = os.path.join(root_dir, "E")
        self.G_dir = os.path.join(root_dir, "G")
        self.Y_dir = os.path.join(root_dir, "Y")
        self.image_files = sorted(os.listdir(self.A_dir))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        # Separate transform for Y, grayscale target images
        self.target_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        A_path = os.path.join(self.A_dir, self.image_files[idx])
        E_path = os.path.join(self.E_dir, self.image_files[idx])
        G_path = os.path.join(self.G_dir, self.image_files[idx])
        Y_path = os.path.join(self.Y_dir, self.image_files[idx])

        # Load images
        A = Image.open(A_path).convert("RGB")
        E = Image.open(E_path).convert("RGB")
        G = Image.open(G_path).convert("RGB")
        Y = Image.open(Y_path).convert("L")  # Grayscale for target

        # Assert to check image size before transform (optional and can be removed in production)
        assert A.size == (140, 140), f"Input image A size must be 140x140, got {A.size}"
        assert E.size == (140, 140), f"Input image E size must be 140x140, got {E.size}"
        assert G.size == (140, 140), f"Input image G size must be 140x140, got {G.size}"
        assert Y.size == (140, 140), f"Target image Y size must be 140x140, got {Y.size}"

        # Apply transforms (resize to 64x64 and normalization)
        A = self.transform(A)
        E = self.transform(E)
        G = self.transform(G)
        Y = self.target_transform(Y)

        # Concatenate the input tensors along channel dimension
        input_tensor = torch.cat([A, E, G], dim=0)

        return input_tensor, Y
