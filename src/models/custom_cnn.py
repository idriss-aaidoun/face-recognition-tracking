"""Small randomly initialized CNN for closed-set face identification.

Use the same RGB preprocessing and identity mapping as the ResNet18 experiment.
No data loading, normalization, pretrained weights, or downloads happen here.
"""

from numbers import Real

from torch import Tensor, nn


class CustomCNN(nn.Module):
    """Four convolution blocks followed by a spatially pooled identity head.

    Each block applies Conv2d(3x3), BatchNorm2d, ReLU, and MaxPool2d(2x2).
    Channels grow from 3 to 32, 64, 128, and 256. At the planned 128x128 input
    size, the final feature map is 256x8x8. Adaptive pooling retains a 4x4 grid
    before dropout and a linear head, preserving coarse spatial information.

    All parameters use PyTorch's default random initialization and are trainable.
    During train(), BatchNorm statistics update and dropout is active; eval()
    freezes the statistics and disables dropout. Outputs are logits, not softmax
    probabilities. Seed PyTorch before construction for reproducible initialization.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be an integer greater than or equal to 2")
        if num_classes < 2:
            raise ValueError("num_classes must be greater than or equal to 2")
        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise TypeError("dropout must be a real number in [0, 1)")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        blocks = []
        in_channels = 3
        for out_channels in (32, 64, 128, 256):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_channels * 4 * 4, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Map preprocessed RGB images [B, 3, H, W] to identity logits [B, K].

        The comparison uses 128x128 images. Other resolutions of at least 64x64
        also leave a feature map large enough for the 4x4 adaptive pooling grid.
        """
        return self.classifier(self.pool(self.features(images)))


def build_custom_cnn(num_classes: int, dropout: float = 0.3) -> CustomCNN:
    """Create a fully trainable CNN from scratch with a configurable class count."""
    return CustomCNN(num_classes=num_classes, dropout=dropout)
