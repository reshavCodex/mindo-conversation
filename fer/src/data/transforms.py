"""
Image transformation pipeline for FER2013.

This module contains all preprocessing and augmentation
used during training and evaluation.
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ----------------------------------------------------------
# ImageNet statistics (required for EfficientNet)
# ----------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(cfg):
    """
    Training transforms with light augmentation.
    """

    size = cfg.model.input_size
    aug = cfg.augmentation

    return transforms.Compose([
        transforms.Resize(
            (size, size),
            interpolation=InterpolationMode.BILINEAR,
        ),

        transforms.RandomHorizontalFlip(
            p=aug.horizontal_flip_prob,
        ),

        transforms.RandomRotation(
            degrees=aug.rotation_degrees,
        ),

        transforms.ColorJitter(
            brightness=aug.brightness,
            contrast=aug.contrast,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_eval_transforms(cfg):
    """
    Validation/Test transforms.
    No augmentation is applied.
    """

    size = cfg.model.input_size

    return transforms.Compose([
        transforms.Resize(
            (size, size),
            interpolation=InterpolationMode.BILINEAR,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])