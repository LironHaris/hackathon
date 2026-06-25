import torch
import torch.nn as nn

# --- Model Hyperparameters ---
KERNEL_SIZE = 3  # Standardized to 5x5 for deeper networks
PADDING = 1
DROPOUT = 0.3
HIDDEN_SIZE_MLP = 256  # Increased slightly to handle 20 classes better


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers (Conv -> BN -> ReLU).
    A skip connection feeds the block's input forward and adds it to the
    output, so each layer also receives the previous input. When the channel
    count changes, the skip path is projected with a 1x1 convolution so the
    shapes match before the addition.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # Project the identity when in/out channels differ, otherwise pass through.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save the input for the skip connection (projected if channels differ).
        identity = self.shortcut(x)

        # Layer 1:  conv -> BN -> ReLU   (ReLU #1 is here)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Layer 2:  conv -> BN  (NO ReLU yet -- it comes after the skip add)
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection: add the input back, THEN apply ReLU  (ReLU #2 is here)
        out = out + identity
        out = self.relu(out)

        return out


class ModelArchitecture(nn.Module):
    """
    Student model architecture with residual blocks, Batch Normalization,
    Dropout, and pooling for robust feature extraction and scaling.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Feature extractor with 4 residual blocks (each = 2 conv layers + skip)
        self.features = nn.Sequential(
            # Block 1
            ResidualBlock(3, 16),
            nn.MaxPool2d(2),

            # Block 2
            ResidualBlock(16, 32),
            nn.MaxPool2d(2),

            # Block 3
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),

            # Block 4 (Crucial for 224x224 images)
            ResidualBlock(64, 128),
            nn.MaxPool2d(4),
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