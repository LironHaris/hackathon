import torch
import torch.nn as nn

# --- Model Hyperparameters ---
KERNEL_SIZE = 5
PADDING = 1
DROPOUT = 0.3
HIDDEN_SIZE_MLP = 128

class ModelArchitecture(nn.Module):
    """
    Student model architecture.

    Students should define their model here.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            flatten_dim = self.features(torch.zeros(1, 3, 224, 224)).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, HIDDEN_SIZE_MLP),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_MLP, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: batch of images

        Returns:
            logits for 20 classes
        """
        x = self.features(x)
        logits = self.classifier(x)
        return logits