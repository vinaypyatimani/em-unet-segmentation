import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class EMDataset(Dataset):
    def __init__(self, image_dir, labels_dir, file_list):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(labels_dir)
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_files = self.images[idx]
        label_files = self.labels[idx]
        img_path = os.path.join(self.image_dir, image_files)
        labels_path = os.path.join(self.labels_dir, label_files)

        image = Image.open(img_path).convert("L")
        label = Image.open(labels_path).convert("L")

        image = np.array(image, dtype=np.float32) / 255.0
        label = np.array(label, dtype=np.float32) / 255.0

        image = torch.tensor(image).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)

        return image, label
