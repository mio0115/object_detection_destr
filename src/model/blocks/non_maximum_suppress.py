from collections.abc import Iterable
import math

import torch
from torch import nn

from ...utils.misc import make_grid
from ...utils.bbox_utils import from_cxcyhw_to_xyxy, get_iou, gen_default_boxes


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
        self._default_boxes = gen_default_boxes(
            shapes=[38, 19, 10, 5, 3, 1],
            scales=self._scale,
            aspect_ratios=self._aspect_ratios,
        )

    def forward(self, features: list[torch.Tensor]):
        # shape of features[0]: (bs, h, w, boxes, d_model)

        pred_boxes, pred_conf = [], []
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
            pred_conf.append(conf.flatten(1, 3))
        all_boxes_with_conf = torch.cat(pred_boxes, 1)
        all_conf = torch.cat(pred_conf, 1)

        sel_boxes, sel_conf = [], []
        for b_bwc, b_c in zip(all_boxes_with_conf, all_conf):
            conf, ind = torch.sort(b_bwc[..., -1], descending=True)
            # drop indeces whose confidence lower than threshold(0.5)
            fil_ind = ind[conf >= 0.5]
            # reorder boxes based on fil_ind and drop their conf
            boxes_cxcyhw = b_bwc[fil_ind, :-1]
            boxes_xyxy = from_cxcyhw_to_xyxy(boxes_cxcyhw)
            tmp_conf = b_c[fil_ind]

            # drop the iou about self
            iou_masks = (
                get_iou(bbox1=boxes_xyxy, bbox2=boxes_xyxy).triu(1) < 0.5
            )
            iou_masks = iou_masks.all(0)

            sel_boxes.append(boxes_cxcyhw[iou_masks])
            sel_conf.append(tmp_conf[iou_masks])

        return sel_boxes, sel_conf


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
    [38, 19, 10, 5, 3, 1]
    nms(
        [
            torch.rand((2, 38, 38, 4, 4 + 20 + 1)),
            torch.rand((2, 19, 19, 6, 4 + 20 + 1)),
            torch.rand((2, 10, 10, 6, 4 + 20 + 1)),
            torch.rand((2, 5, 5, 6, 4 + 20 + 1)),
            torch.rand((2, 3, 3, 4, 4 + 20 + 1)),
            torch.rand((2, 1, 1, 4, 4 + 20 + 1)),
        ]
    )
