import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Model Hyperparameters ---
KERNEL_SIZE = 5  # Standardized to 5x5 for deeper networks
PADDING = 2
DROPOUT = 0.3
HIDDEN_SIZE_MLP = 512  # Increased slightly to handle 20 classes better


class ModelArchitecture(nn.Module):
    """
    Student model architecture with Batch Normalization, Dropout,
    and Adaptive Pooling for robust feature extraction and scaling.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()
        # Feature extractor with 4 Convolutional blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(64),  # Fixed: matches output channels
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4 (Crucial for 224x224 images)
            nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 5
            nn.Conv2d(128, 128, kernel_size=KERNEL_SIZE, padding=PADDING,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 128 channels * 7 * 7 = 6272 (28x28 after block 3, then MaxPool2d(4) → 7x7)
        flatten_dim = 128 * 7 * 7

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, HIDDEN_SIZE_MLP),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(HIDDEN_SIZE_MLP, num_classes)
        )

    def _resize_to_square(self, x: torch.Tensor, target_size: int = 224) -> torch.Tensor:
        """
        Pads a [B, 3, H, W] batch to a square (preserving aspect ratio, no
        distortion) and resizes it to (target_size, target_size), so the
        model can accept input images of any size.
        """
        _, _, h, w = x.shape
        max_wh = max(h, w)
        p_left = (max_wh - w) // 2
        p_right = max_wh - w - p_left
        p_top = (max_wh - h) // 2
        p_bottom = max_wh - h - p_top

        x = F.pad(x, (p_left, p_right, p_top, p_bottom), mode='constant', value=0)
        if max_wh != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): A batch of input images, shape [B, 3, H, W]
                for any H, W.

        Returns:
            torch.Tensor: The unnormalized logits for the 20 classes.
        """
        x = self._resize_to_square(x)
        x = self.features(x)
        logits = self.classifier(x)
        return logits