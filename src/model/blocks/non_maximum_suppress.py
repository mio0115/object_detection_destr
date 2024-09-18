from collections.abc import Iterable
import math

import torch
from torch import nn

from ...utils.misc import make_grid
from ...utils.bbox_utils import from_cxcyhw_to_xyxy, get_iou


class NonMaximumSuppress(nn.Module):
    def __init__(
        self,
        scale: Iterable[float],
        aspect_ratios: Iterable[float],
        *args,
        **kwargs
    ):
        super(NonMaximumSuppress, self).__init__(*args, **kwargs)

        # different scale for different size of feature map
        self._scale = scale
        # 4 boxes are {1, 1, 2, 1/2}
        # 6 boxes are {1, 1, 2, 3, 1/2, 1/3}
        self._aspect_ratios = aspect_ratios

        assert (
            len(self._scale) == len(self._aspect_ratios) + 1
        ), "length of scale should equal to length of aspect ratio + 1"

        # the shape is specified when we use vgg16 as backbone, need to change if we swap it out
        self._default_boxes = self._gen_default_boxes(
            shapes=[38, 19, 10, 5, 3, 1],
            scales=self._scale,
            aspect_ratios=self._aspect_ratios,
        )

    def forward(self, features: list[torch.Tensor]):
        # shape of features[0]: (bs, h, w, boxes, d_model)

        pred_boxes = []
        device = features[0].device

        for def_box, ft in zip(self._default_boxes, features):
            coord, conf = ft[..., :4], ft[..., 4:].softmax(-1)

            box_coord = torch.stack(
                [
                    def_box[..., 0] + def_box[..., 3] * coord[..., 0],
                    def_box[..., 1] + def_box[..., 2] * coord[..., 1],
                    def_box[..., 2] * coord[..., 2].exp(),
                    def_box[..., 3] * coord[..., 3].exp(),
                ],
                -1,
            )
            conf_wo_dummy, _ = conf[..., :-1].max(-1, keepdim=True)
            box_with_conf = torch.cat([box_coord, conf_wo_dummy], -1).flatten(
                1, 3
            )

            pred_boxes.append(box_with_conf)
        all_boxes_with_conf = torch.cat(pred_boxes, 1)

        all_boxes = []
        for b_bwc in all_boxes_with_conf:
            conf, ind = torch.sort(b_bwc[..., -1], descending=True)
            # drop indeces whose confidence lower than threshold(0.5)
            fil_ind = ind[conf >= 0.5]
            # reorder boxes based on fil_ind and drop their conf
            boxes_cxcyhw = b_bwc[fil_ind, :-1]
            boxes_xyxy = from_cxcyhw_to_xyxy(boxes_cxcyhw)

            # drop the iou about self
            ious = get_iou(bbox1=boxes_xyxy, bbox2=boxes_xyxy) - torch.eye(
                n=boxes_xyxy.size(0), device=device
            )

            num_boxes, _ = boxes_xyxy.shape
            preserved_mask = torch.ones(
                size=num_boxes, dtype=torch.bool, device=device
            )
            for ind, iou in enumerate(ious):
                preserved_mask &= iou[ind:] < 0.5

            all_boxes.append(boxes_cxcyhw[preserved_mask])

        return all_boxes

    def _gen_default_boxes(
        self,
        shapes: Iterable[int],
        scales: Iterable[float],
        aspect_ratios: Iterable[float],
    ):
        default_boxes = []

        for ind, (shape, ar) in enumerate(zip(shapes, aspect_ratios)):
            num_boxes = (len(ar) + 1) * 2
            centers = (
                make_grid(height=shape, width=shape, bias=0.5, norm=True)
                .unsqueeze(0)
                .repeat(num_boxes, 1, 1, 1)
            )

            scale = scales[ind]
            g_scale = math.sqrt(scales[ind] * scales[ind + 1])

            hw_pairs = [(scale, scale), (g_scale, g_scale)]
            for ar_ in ar:
                sqrt_ar = math.sqrt(ar_)
                hw_pairs.append((scale * sqrt_ar, scale / sqrt_ar))
                hw_pairs.append((scale / sqrt_ar, scale * sqrt_ar))
            hw_pairs = torch.tensor(hw_pairs)[:, None, None, :].repeat(
                1, shape, shape, 1
            )

            default_boxes.append(
                torch.cat([centers, hw_pairs], -1).unsqueeze(0)
            )

        return default_boxes


def build_nms(args):
    one_step = (args.scale_max - args.scale_min) / 5

    nms = NonMaximumSuppress(
        scale=torch.arange(
            start=args.scale_min,
            end=args.scale_max + one_step + 0.01,
            step=one_step,
            dtype=torch.float32,
            device=args.device,
        ),
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    ).to(args.device)

    return nms


if __name__ == "__main__":
    scale_min, scale_max = 0.2, 0.9
    one_step = (scale_max - scale_min) / 5

    nms = NonMaximumSuppress(
        scale=torch.arange(
            start=scale_min,
            end=scale_max + one_step + 0.01,
            step=one_step,
            dtype=torch.float32,
        ),
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    )
