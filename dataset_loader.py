import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ForgeryDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_paths = sorted(os.listdir(image_folder))
        self.mask_paths = sorted(os.listdir(mask_folder))
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_paths[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_paths[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return image, mask

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = ForgeryDataset('dataset/images', 'dataset/masks', transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Dataset Loaded Successfully!")
