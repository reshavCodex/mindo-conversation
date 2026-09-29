"""
EfficientNet-B0 model definition for FER2013.

This module builds the model and provides helper functions
to freeze and unfreeze different parts of the network during
training.
"""

import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


def build_model(cfg):
    """
    Create EfficientNet-B0 and replace the classifier.
    """

    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if cfg.model.pretrained
        else None
    )

    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        cfg.model.num_classes,
    )

    return model


# ----------------------------------------------------------
# Fine-tuning helpers
# ----------------------------------------------------------

def freeze_backbone(model):
    """
    Freeze every layer except the classifier.
    """

    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_stage(model):
    """
    Unfreeze only the final EfficientNet feature stage.
    """

    # keep everything frozen
    for param in model.features.parameters():
        param.requires_grad = False

    # unfreeze final MBConv block
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # classifier always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    """
    Train the entire network.
    """

    for param in model.parameters():
        param.requires_grad = True


def count_trainable_parameters(model):
    """
    Return number of trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )