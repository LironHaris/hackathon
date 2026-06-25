import torch
import torch.nn as nn

# --- Model Hyperparameters ---
KERNEL_SIZE = 3  # Standardized to 3x3 for deeper networks
PADDING = 1
DROPOUT = 0.3
HIDDEN_SIZE_MLP = 256  # Increased slightly to handle 20 classes better


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

            # Adaptive Pooling forces the spatial dimensions to exactly 4x4
            # regardless of input image size, eliminating massive linear layers.
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # 128 channels * 7 * 7 = 6272 (14x14 after 4 MaxPool2d(2), 14/7=2 is MPS-compatible)
        flatten_dim = 128 * 7 * 7

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, HIDDEN_SIZE_MLP),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(HIDDEN_SIZE_MLP, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): A batch of input images.

        Returns:
            torch.Tensor: The unnormalized logits for the 20 classes.
        """
        x = self.features(x)
        logits = self.classifier(x)
        return logits