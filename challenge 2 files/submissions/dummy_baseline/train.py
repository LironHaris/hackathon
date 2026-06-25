#!/usr/bin/env python3

from pathlib import Path

import joblib

from model import ModelArchitecture
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class SquarePad:
    """
    A custom PyTorch transform that pads an image to a square.
    It adds symmetrical black borders to the shorter edge so that width == height,
    preventing aspect ratio distortion during resizing.
    """

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)

        # Calculate padding for left/right and top/bottom
        p_left = int((max_wh - w) / 2)
        p_top = int((max_wh - h) / 2)
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top

        # Pad with 0 (black)
        return TF.pad(image, (p_left, p_top, p_right, p_bottom), 0, 'constant')


def get_transforms():
    """
    Returns separate torchvision augmentation pipelines for training and validation.
    """
    val_transform = T.Compose([
        SquarePad(),  # 1. Pad to square with black borders
        T.Resize((224, 224)),  # 2. Resize the square to 224x224
        T.ToTensor(),  # 3. Convert PIL Image to PyTorch Tensor (C, H, W) & scales to [0, 1]
        T.Normalize(  # 4. Normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_transform = T.Compose([
        SquarePad(),
        T.Resize((224, 224)),
        # TODO: Add specific training augmentations here in the future
        # e.g., T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.2),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


class RobustObjectDataset(Dataset):
    """
    A PyTorch Dataset that lazily loads images from a DataFrame using PIL.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

        if self.transform is None:
            # Fallback to validation transforms if none provided
            _, self.transform = get_transforms()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = row['label']

        try:
            # Use PIL instead of OpenCV.
            # .convert('RGB') ensures consistency even if some images are grayscale or have an alpha channel.
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image at {image_path}. Error: {e}")

        # torchvision transforms take the image directly and return the final tensor
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

"""

DATA_ROOT = Path("../../dataset")
OUTPUT = Path("weights.joblib")

def main() -> None:
    
    #Dummy training script.

    #This does not train.
    #It only creates the dummy CNN and saves its zero-initialized weights.
    #The resulting model should always predict class 0.
    

    model = ModelArchitecture(num_classes=20)

    state_dict = model.state_dict()
    joblib.dump(state_dict, OUTPUT)

    print(f"Saved dummy weights to {OUTPUT}")
    print("This dummy CNN always predicts class 0.")


if __name__ == "__main__":
    main()"""