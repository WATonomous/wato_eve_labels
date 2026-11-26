"""SegFormer-B0 through segmentation-models-pytorch.

B0 because we need 8 cameras inside a ~5 ms budget on a 3090, and because the
fine-tune set is small: a bigger encoder would memorise our 500 frames. The
encoder is ImageNet-pretrained, everything is fully convolutional so the same
weights run on 512x512 crops (public-data training) and 1280x1024 panos
(inference) without changing anything.
"""
import segmentation_models_pytorch as smp
import torch.nn as nn

from ..classes import NUM_CLASSES


def build_model(arch="mit_b0", num_classes=NUM_CLASSES, encoder_weights="imagenet"):
    return smp.Segformer(
        encoder_name=arch,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6
