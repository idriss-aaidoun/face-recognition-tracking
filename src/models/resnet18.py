"""ResNet18 identity logits with automatic evaluation mode for frozen blocks.

Preprocessing belongs to the data pipeline. Importing this module never downloads
weights; explicitly building with ``pretrained=True`` may download ImageNet weights.
"""

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


def _validate_backbone_mode(mode: str) -> None:
    if mode not in ("none", "layer4", "all"):
        raise ValueError(
            "trainable_backbone must be 'none', 'layer4', or 'all'; "
            f"got {mode!r}"
        )


class ResNet18Classifier(nn.Module):
    """Wrap a torchvision network and preserve frozen blocks across train/eval.

    The torchvision model, including its replacement ``fc``, is available as
    ``network``. Use this wrapper's forward/train/eval methods, rather than calling
    ``network.train()`` directly. State-dict keys have the ``network.`` prefix.
    """

    def __init__(self, network: nn.Module, trainable_backbone: str) -> None:
        super().__init__()
        self.network = network
        self.set_trainable_backbone(trainable_backbone)

    @property
    def trainable_backbone(self) -> str:
        """Return the current parameter-freezing policy."""
        return self._trainable_backbone

    def _child_is_trainable(self, name: str) -> bool:
        return (
            name == "fc"
            or self.trainable_backbone == "all"
            or (self.trainable_backbone == "layer4" and name == "layer4")
        )

    def set_trainable_backbone(self, mode: str) -> None:
        """Change the policy in place, retaining weights and current train/eval mode.

        Rebuild the optimizer after a policy change so it includes newly trainable
        parameters. Stale gradients on newly frozen parameters are cleared.
        """
        _validate_backbone_mode(mode)
        self._trainable_backbone = mode
        for name, child in self.network.named_children():
            trainable = self._child_is_trainable(name)
            for parameter in child.parameters():
                parameter.requires_grad_(trainable)
                if not trainable:
                    parameter.grad = None
        self.train(self.training)

    def train(self, mode: bool = True) -> "ResNet18Classifier":
        """Keep frozen children in eval mode, including their BatchNorm buffers.

        In ``layer4`` mode only layer4 and fc follow the requested training mode.
        In ``all`` mode all children follow it. Calling eval() puts all children
        in evaluation mode without changing which parameters require gradients.
        """
        super().train(mode)
        if mode:
            for name, child in self.network.named_children():
                if not self._child_is_trainable(name):
                    child.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Return unnormalized identity logits of shape [batch_size, num_classes]."""
        return self.network(images)


def build_resnet18(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone: str = "none",
) -> ResNet18Classifier:
    """Build a fixed-identity classifier with a randomly initialized new head.

    ``none`` trains only fc; ``layer4`` trains layer4 and fc; ``all`` trains all
    parameters. Frozen blocks also retain their BatchNorm running statistics.
    Use ``pretrained=False`` for offline technical checks, not recognition scores.
    Weight-loading errors propagate: pretrained weights never silently fall back
    to random initialization. Validate arguments before any possible download.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer greater than or equal to 2")
    if num_classes < 2:
        raise ValueError("num_classes must be greater than or equal to 2")
    if not isinstance(pretrained, bool):
        raise TypeError("pretrained must be a bool")
    _validate_backbone_mode(trainable_backbone)

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return ResNet18Classifier(model, trainable_backbone)
