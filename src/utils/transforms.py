from enum import Enum

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
                v2.ToTensor(),
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.RandomRotation(20),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # trans_type is VALID or TEST
        trans_fn = v2.compose(
            [
                v2.ToTensor(),
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return trans_fn
