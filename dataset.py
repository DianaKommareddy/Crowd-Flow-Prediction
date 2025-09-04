import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CrowdFlowDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to dataset root folder with subfolders A/, E/, G/, Y/
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get sorted file lists from each folder
        self.A_files = sorted(os.listdir(os.path.join(root_dir, "A")))
        self.E_files = sorted(os.listdir(os.path.join(root_dir, "E")))
        self.G_files = sorted(os.listdir(os.path.join(root_dir, "G")))
        self.Y_files = sorted(os.listdir(os.path.join(root_dir, "Y")))

        # Ensure all folders have equal number of files
        assert len(self.A_files) == len(self.E_files) == len(self.G_files) == len(self.Y_files), \
            "Mismatch in number of files between A, E, G, and Y"

    def __len__(self):
        return len(self.A_files)

    def __getitem__(self, idx):
        # File paths
        A_path = os.path.join(self.root_dir, "A", self.A_files[idx])
        E_path = os.path.join(self.root_dir, "E", self.E_files[idx])
        G_path = os.path.join(self.root_dir, "G", self.G_files[idx])
        Y_path = os.path.join(self.root_dir, "Y", self.Y_files[idx])

        # Open images as grayscale
        A_img = Image.open(A_path).convert("L")  # "L" = grayscale
        E_img = Image.open(E_path).convert("L")
        G_img = Image.open(G_path).convert("L")
        Y_img = Image.open(Y_path).convert("L")

        # Apply transform if given
        if self.transform:
            A_img = self.transform(A_img)
            E_img = self.transform(E_img)
            G_img = self.transform(G_img)
            Y_img = self.transform(Y_img)

        return A_img, E_img, G_img, Y_img


# Default transform (1-channel grayscale, resized, tensor)
default_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensures 1 channel
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Example usage:
# dataset = CrowdFlowDataset(root_dir="Test Dataset", transform=default_transform)
