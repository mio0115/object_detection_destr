from collections.abc import Iterable

import torch
from torch import nn

from ...utils.bbox_utils import gen_centers, get_iou


class NonMaximumSuppress(nn.Module):
    def __init__(
        self,
        scale: Iterable[float],
        aspect_ratios: Iterable[float],
        origin_image_size: Iterable[int, int],
        *args,
        **kwargs
    ):
        super(NonMaximumSuppress, self).__init__(*args, **kwargs)

        self._scale = scale
        self._aspect_ratios = aspect_ratios
        # (height, width)
        self._org_img_size = origin_image_size
        # shape of default boxes (1, height, width, boxes, 4)
        self._default_boxes = self._build_default_boxes()

    def forward(self, features: list[torch.Tensor]):
        pred_boxes = []

        for def_box, ft in zip(self._default_boxes, features):
            bs, h, w, boxes, d_model = ft.shape
            coord, conf = ft[..., :4], ft[..., 4:]

            pred_coord = torch.stack(
                [
                    def_box[..., 0] + def_box[..., 3] * coord[..., 0],
                    def_box[..., 1] + def_box[..., 2] * coord[..., 1],
                    def_box[..., 2] * coord[..., 2].exp(),
                    def_box[..., 3] * coord[..., 3].exp(),
                ],
                -1,
            )
            pred_coord[..., 0::2] *= w / self._org_img_size[1]
            pred_coord[..., 1::2] *= h / self._org_img_size[0]

            pred_boxes.append(pred_coord)
