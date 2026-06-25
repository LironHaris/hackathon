#!/usr/bin/env python3

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

from model import ModelArchitecture

# --- Training Configurations ---
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 2
_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(_DIR, "train_split.csv")
VAL_CSV = os.path.join(_DIR, "val_split.csv")


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SquarePad:
    """
    A custom PyTorch transform that pads an image to a square.
    It adds symmetrical black borders to the shorter edge so that width == height,
    preventing aspect ratio distortion during resizing.
    """

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = int((max_wh - w) / 2)
        p_top = int((max_wh - h) / 2)
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        return TF.pad(image, (p_left, p_top, p_right, p_bottom), 0, 'constant')


def get_transforms():
    """
    Returns separate torchvision augmentation pipelines for training and validation.
    """
    val_transform = T.Compose([
        SquarePad(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = T.Compose([
        SquarePad(),
        T.Resize((224, 224)),
        # Add future specific training augmentations here
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


class RobustObjectDataset(Dataset):
    """
    A PyTorch Dataset that lazily loads images from a DataFrame using PIL.
    """

    def __init__(self, df: pd.DataFrame, transform=None, base_dir: str = None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform
        if self.transform is None:
            _, self.transform = get_transforms()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        image_path = row['image_path']
        if self.base_dir and not os.path.isabs(image_path):
            image_path = os.path.normpath(os.path.join(self.base_dir, image_path))
        label = row['label']

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image at {image_path}. Error: {e}")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(train_csv_path: str, val_csv_path: str, batch_size: int, num_workers: int):
    """
    Reads the CSV files and returns PyTorch DataLoaders.
    """
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    train_transform, val_transform = get_transforms()

    train_dataset = RobustObjectDataset(df=df_train, transform=train_transform, base_dir=_DIR)
    val_dataset = RobustObjectDataset(df=df_val, transform=val_transform, base_dir=_DIR)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


class ModelTrainer:
    """
    A unified class to handle the training and validation loops for a PyTorch model.
    Tracks losses and accuracy, and automatically saves the best performing model.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device,
                 num_epochs: int, save_dir: str = "checkpoints"):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train(self) -> dict:
        best_val_acc = 0.0
        print(f"Starting training on device: {self.device} for {self.num_epochs} epochs...")

        for epoch in range(1, self.num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.num_epochs} ---")
            train_loss = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"🌟 New best model saved! (Accuracy: {best_val_acc:.2f}%)")

        print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc:.2f}%")
        return self.history

    def _train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return running_loss / len(self.train_loader.dataset)

    def _validate_epoch(self) -> tuple:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc


def plot_training_results(history: dict, save_dir: str = "plots"):
    """
    Plots the training and validation metrics (Loss and Accuracy) and saves the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', color='red', marker='o')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', color='green', marker='o')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_results.png")
    plt.savefig(plot_path)
    print(f"\nTraining plots saved to: {plot_path}")
    plt.show()


def main():
    # Enforce reproducibility
    seed_everything(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing on device: {device}")

    print("\nInitializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        train_csv_path=TRAIN_CSV,
        val_csv_path=VAL_CSV,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print("\nInitializing Model...")
    model = ModelArchitecture(num_classes=20)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training Pipeline...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        save_dir="checkpoints"
    )

    history = trainer.train()

    plot_training_results(history)


if __name__ == '__main__':
    main()