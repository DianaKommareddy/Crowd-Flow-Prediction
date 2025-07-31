# utils.py

import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class CrowdDataset(Dataset):
    def __init__(self, root_dir, config):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(os.path.join(root_dir, "A")))
        self.transform = transforms.Compose([
            transforms.Resize((config.img_height, config.img_width)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img_A = self.transform(Image.open(os.path.join(self.root_dir, "A", name)).convert("L"))
        img_E = self.transform(Image.open(os.path.join(self.root_dir, "E", name)).convert("L"))
        img_G = self.transform(Image.open(os.path.join(self.root_dir, "G", name)).convert("L"))
        label = self.transform(Image.open(os.path.join(self.root_dir, "Y", name)).convert("L"))

        # Stack A, E, G as 3 channels
        input_img = torch.cat([img_A, img_E, img_G], dim=0)
        return input_img, label
