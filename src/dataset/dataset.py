import torch
import torchvision as tv
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torchvision.transforms import ToTensor

from ..utils.bbox_utils import from_xywh_to_xyxy, filter_flat_box
from .transforms import TransformTypes


class WiderFace(tv.datasets.WIDERFace):
    def __init__(
        self,
        root,
        split: str | TransformTypes,
        transform,
        max_items_per_img: int = 300,
        augment_factor: float = 10,
    ):
        if isinstance(split, TransformTypes):
            super(WiderFace, self).__init__(root, split.value)
        elif isinstance(split, str) and (split in ["train", "val", "test"]):
            super(WiderFace, self).__init__(root, split)
        else:
            raise ValueError(f"unknown split: {split}")

        self._transforms = transform
        self._augment_factor = augment_factor
        self._max_items_per_image = max_items_per_img
        self._to_tensor = ToTensor()

    def __len__(self) -> int:
        return super().__len__() * self._augment_factor

    def __getitem__(self, index: int) -> torch.Tuple[torch.Any]:
        img, targets = super().__getitem__(index % super().__len__())

        _, indices = torch.sort(
            targets["bbox"][..., 2] * targets["bbox"][..., 3], descending=True
        )
        bboxes = targets["bbox"].index_select(index=indices, dim=0)

        bboxes = BoundingBoxes(
            from_xywh_to_xyxy(bboxes),
            format=BoundingBoxFormat.XYXY,
            canvas_size=self._to_tensor(img).shape[-2:],
            dtype=torch.float32,
        )
        new_img, new_bboxes = self._transforms(img, bboxes)

        new_bboxes = filter_flat_box(new_bboxes)

        new_bboxes = new_bboxes[: self._max_items_per_image]

        # normalize
        new_bboxes[:, 0::2] /= new_img.size(2)  # width
        new_bboxes[:, 1::2] /= new_img.size(1)  # height

        # one-hot label
        # new_labels = torch.tensor([[1.0, 0.0]]).expand(size=(new_bboxes.size(0), 2))
        new_labels = torch.zeros(size=(new_bboxes.size(0),)).long()

        return new_img, {"boxes": new_bboxes, "labels": new_labels}


def widerface_collate_fn(batch):
    img, targets = zip(*batch)

    img = torch.stack(img)

    return img, targets
