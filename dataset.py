import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CrowdFlowDataset(Dataset):
    def __init__(self, root_dir):
        self.agent_dir = os.path.join(root_dir, 'A')
        self.env_dir = os.path.join(root_dir, 'E')
        self.goal_dir = os.path.join(root_dir, 'G')
        self.gt_dir = os.path.join(root_dir, 'Y')
        self.filenames = sorted(os.listdir(self.agent_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        def load_grayscale(path):
            img = Image.open(path).convert('L')
            img = img.resize((140, 140))  # Inputs and GT are both 140x140
            return np.array(img, dtype=np.float32) / 255.0

        agent = load_grayscale(os.path.join(self.agent_dir, filename))
        env = load_grayscale(os.path.join(self.env_dir, filename))
        goal = load_grayscale(os.path.join(self.goal_dir, filename))
        gt = load_grayscale(os.path.join(self.gt_dir, filename))

        # Stack inputs to form (9, 140, 140)
        input_tensor = torch.tensor(np.stack([agent, env, goal] * 3), dtype=torch.float32)
        gt_tensor = torch.tensor(gt[np.newaxis, :, :], dtype=torch.float32)  # (1, 140, 140)

        return input_tensor, gt_tensor
