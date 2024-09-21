from enum import Enum
from typing import Optional
import math

import torch
import numpy as np
from torch import nn
from torchvision.transforms import v2

from ..utils.bbox_utils import (
    BBoxType,
    from_xyxy_to_cxcyhw,
    from_cxcyhw_to_xyxy,
    update_coord_new_boundary,
)


class TransformTypes(Enum):
    TRAIN = "train"
    VALID = "val"
    TEST = "test"


class TransBoxCoord(nn.Module):
    def __init__(
        self,
        norm: bool = False,
        from_type: Optional[BBoxType] = None,
        to_type: Optional[BBoxType] = None,
    ):
        self._norm = norm
        self._from_type = from_type
        self._to_type = to_type

        match (self._from_type, self._to_type):
            case (BBoxType.XYXY, BBoxType.CXCYHW):
                self._trans_fn = from_xyxy_to_cxcyhw
            case (BBoxType.CXCYHW, BBoxType.XYXY):
                self._trans_fn = from_cxcyhw_to_xyxy
            case _:
                self._trans_fn = None

    def forward(self, img, boxes, labels):
        h, w = img.shape[1:]
        if self._norm:
            boxes = torch.stack(
                [
                    boxes[..., 0] / w,
                    boxes[..., 1] / h,
                    boxes[..., 2] / w,
                    boxes[..., 3] / h,
                ],
                0,
            )

        if self._trans_fn:
            boxes = self._trans_fn(boxes)

        return img, boxes, labels


class RandomPatchWithIoUBound(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RandomPatchWithIoUBound, self).__init__(*args, **kwargs)

        self._sample_options = (
            -1,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            0,
        )

    def forward(self, img, boxes, labels):
        # expect the boxes are in cxcyhw format
        h, w = img.shape[1:]
        num_boxes = boxes.shape[0]

        while True:
            min_obj_remain = np.random.choice(self._sample_options)
            if min_obj_remain < 0:
                # if mode is -1, we do not crop the image
                return img, boxes, labels

            min_num_boxes = int(min_obj_remain * num_boxes)

            for _ in range(50):
                # get the range of new image
                scale = np.random.uniform(0.1, 1)
                ar = np.random.uniform(0.5, 2)

                patch_h = min(int(np.floor(np.sqrt(scale / ar)) * h), h)
                patch_w = min(int(np.floor(np.sqrt(scale * ar)) * w), w)

                # compute the coordinates on the original image
                min_x = np.random.randint(0, w - patch_w)
                min_y = np.random.randint(0, h - patch_h)
                max_x = min_x + patch_h
                max_y = min_y + patch_w

                # drop objects not in the new image
                x_mask = (boxes[..., 0] >= min_x) & (boxes[..., 0] <= max_x)
                y_mask = (boxes[..., 1] >= min_y) & (boxes[..., 1] <= max_y)
                in_patch = x_mask & y_mask
                if np.count_nonzero(in_patch) >= min_num_boxes:
                    in_boxes = update_coord_new_boundary(
                        boxes[in_patch],
                        new_boundary=(min_x, min_y, max_x, max_y),
                    )
                    in_labels = labels[in_patch]
                    new_img = img[:, min_y:max_y, min_x:max_x]

                    return new_img, in_boxes, in_labels


def build_transform_ssd(trans_type: TransformTypes):
    trans_fn = None

    if trans_type == TransformTypes.TRAIN:

        trans_fn = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                TransBoxCoord(
                    norm=False,
                    from_type=BBoxType.XYXY,
                    to_type=BBoxType.CXCYHW,
                ),
                RandomPatchWithIoUBound(),
                TransBoxCoord(norm=True),
                v2.Resize(size=(300, 300)),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        trans_fn = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                TransBoxCoord(
                    norm=True, from_type=BBoxType.XYXY, to_type=BBoxType.CXCYHW
                ),
                v2.Resize(size=(300, 300)),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return trans_fn


def build_transform(trans_type: TransformTypes):
    trans_fn = None

    if trans_type == TransformTypes.TRAIN:
        trans_fn = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomResizedCrop(size=640),
                v2.RandomHorizontalFlip(0.5),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        # trans_type is VALID or TEST
        trans_fn = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(672),
                v2.CenterCrop(640),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return trans_fn
