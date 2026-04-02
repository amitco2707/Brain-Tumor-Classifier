import torch.nn as nn
from torchvision import models


def build_model(num_classes=2):
    """
    Loads a pretrained ResNet-18 and adapts it for binary classification.

    Args:
        num_classes: number of output categories (2 = tumor / no tumor)

    Returns:
        model: the modified ResNet-18, ready to be trained
    """

    # Load ResNet-18 with pretrained weights from ImageNet
    # weights=IMAGENET1K_V1 means: use the version trained on ImageNet (1000 classes)
    # This downloads the weights the first time, then caches them locally
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all the existing layers so their weights don't change during training
    # Why? These layers already know how to detect edges, textures, and shapes.
    # We want to preserve that knowledge and only teach the final layer our task.
    # This also makes training much faster — fewer parameters to update.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    # ResNet-18's original final layer: 512 inputs → 1000 outputs (ImageNet classes)
    # Our new final layer:              512 inputs → 2 outputs (tumor / no tumor)
    # in_features grabs the 512 automatically so we don't hardcode it
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # Note: only this new layer has requires_grad=True, so only it will be trained

    return model
