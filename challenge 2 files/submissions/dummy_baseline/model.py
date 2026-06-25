import torch
import torch.nn as nn

# --- Model Hyperparameters ---
KERNEL_SIZE = 5
PADDING = 1
DROPOUT = 0.3
HIDDEN_SIZE_MLP = 128

class ModelArchitecture(nn.Module):
    """
    Student model architecture with Batch Normalization and Dropout
    for improved training stability and robustness against overfitting.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Feature extractor with Convolution, BatchNorm, ReLU, and MaxPool
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(16),  # Normalizes the 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(32),  # Normalizes the 32 output channels
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dynamically calculate the flattened dimension based on input size
        with torch.no_grad():
            flatten_dim = self.features(torch.zeros(1, 3, 224, 224)).numel()

        # Classifier with Dropout for regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, HIDDEN_SIZE_MLP),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),  # Randomly zeroes some elements to prevent overfitting
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