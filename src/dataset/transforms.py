from enum import Enum

import torch
from torchvision.transforms import v2


class TransformTypes(Enum):
    TRAIN = "train"
    VALID = "val"
    TEST = "test"


def build_transform(trans_type: TransformTypes):
    trans_fn = None

    if trans_type == TransformTypes.TRAIN:
        trans_fn = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomResizedCrop(size=224),
                v2.RandomRotation(20),
                v2.RandomHorizontalFlip(0.5),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # trans_type is VALID or TEST
        trans_fn = v2.compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(500),
                v2.CenterCrop(480),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return trans_fn
