import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class DUTSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=128, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def _load_pair(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")   
        return img, mask

    def _augment(self, img, mask):
        
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            img = TF.adjust_brightness(img, factor)

        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load_pair(idx)

        
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        
        if self.augment:
            img, mask = self._augment(img, mask)

        
        img = TF.to_tensor(img)            
        mask = TF.to_tensor(mask)         

       
        mask = (mask > 0.5).float()

        return img, mask


def get_image_mask_paths(root_dir):
    images_dir = os.path.join(root_dir, "images")
    masks_dir = os.path.join(root_dir, "masks")

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    image_paths = []
    mask_paths = []

    for fname in image_files:
        base = os.path.splitext(fname)[0]
        
        for ext in [".png", ".jpg", ".jpeg"]:
            possible = os.path.join(masks_dir, base + ext)
            if os.path.exists(possible):
                image_paths.append(os.path.join(images_dir, fname))
                mask_paths.append(possible)
                break

    print(f"Loaded {len(image_paths)} image-mask pairs.")
    return image_paths, mask_paths


def create_dataloaders(root_dir, batch_size=8, image_size=128, num_workers=0):
    
    image_paths, mask_paths = get_image_mask_paths(root_dir)

    
    train_imgs, temp_imgs, train_msks, temp_msks = train_test_split(
        image_paths, mask_paths, test_size=0.30, random_state=42
    )
    val_imgs, test_imgs, val_msks, test_msks = train_test_split(
        temp_imgs, temp_msks, test_size=0.50, random_state=42
    )

    train_dataset = DUTSDataset(train_imgs, train_msks, image_size, augment=True)
    val_dataset = DUTSDataset(val_imgs, val_msks, image_size, augment=False)
    test_dataset = DUTSDataset(test_imgs, test_msks, image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
