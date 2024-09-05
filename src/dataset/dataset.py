import torch
import torchvision as tv
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from ..utils.bbox_utils import from_xywh_to_xyxy


class WiderFace(tv.datasets.WIDERFace):
    def __init__(
        self,
        root,
        split,
        transform,
        items_per_img: int = 300,
        augment_factor: float = 10,
    ):
        super(WiderFace, self).__init__(root, split)
        self._transforms = transform
        self._augment_factor = augment_factor
        self._items_per_image = items_per_img

    def __len__(self) -> int:
        return super().__len__() * self._augment_factor

    def __getitem__(self, index: int) -> torch.Tuple[torch.Any]:
        img, targets = super().__getitem__(index)

        img = self._to_tesnor(img)

        _, indices = torch.sort(
            targets["bbox"][..., 2] * targets["bbox"][..., 3], descending=True
        )
        bboxes = targets["bbox"].index_select(indices)

        bboxes = BoundingBoxes(
            from_xywh_to_xyxy(bboxes),
            format=BoundingBoxFormat.XYXY,
            canvas_size=img.shape[1:],
            dtype=torch.float32,
        )
        new_img, new_bboxes = self._transforms(img, bboxes)

        new_bboxes = new_bboxes[: self._items_per_image]
        new_labels = torch.Tensor([[1.0, 0.0]]).expand(size=(new_bboxes.size(0), 2))

        pads = self._items_per_image - new_bboxes.size(0)
        new_bboxes = torch.concat(
            [new_bboxes, torch.full(size=(pads, 4), fill_value=-1)], dim=0
        )
        new_labels = torch.concat(
            [new_labels, torch.Tensor([[0.0, 1.0]]).expand(size=(pads, 2))], dim=0
        )

        return new_img, (new_bboxes, new_labels)
