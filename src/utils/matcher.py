# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DESTR (https://github.com/helq2612/destr)
# Copyright (c) 2022 CVPR?. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .bbox_utils import (
    from_cxcyhw_to_xyxy,
    from_xyxy_to_cxcyhw,
    complete_iou,
    gen_default_boxes,
    get_iou,
)
from .misc import to_device


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_ciou: float = 1
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_ciou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_ciou = cost_ciou
        assert (
            self.cost_class != 0 or self.cost_bbox != 0 or self.cost_ciou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries, _ = outputs["pred_class"].shape

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_class"].flatten(0, 1).sigmoid()
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes

        # turn one-hot encoding labels to normal labels
        tgt_ids = torch.cat([tgt["labels"] for tgt in targets]).argmax(-1)
        tgt_bbox = torch.cat([tgt["boxes"] for tgt in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the ciou cost betwen boxes
        cost_ciou = complete_iou(from_cxcyhw_to_xyxy(out_bbox), tgt_bbox)

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_ciou * cost_ciou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(tgt["boxes"]) for tgt in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class HungarianMatcherWoL1(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_ciou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_ciou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_ciou = cost_ciou
        assert self.cost_class != 0 or self.cost_ciou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries, _ = outputs["pred_class"].shape

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_class"].flatten(0, 1).sigmoid()
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([tgt["labels"] for tgt in targets])
        tgt_bbox = torch.cat([tgt["boxes"] for tgt in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the ciou cost betwen boxes
        cost_ciou = complete_iou(from_cxcyhw_to_xyxy(out_bbox), tgt_bbox)

        # Final cost matrix
        C = self.cost_class * cost_class + self.cost_ciou * cost_ciou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(tgt["boxes"]) for tgt in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class SimpleMatcher(nn.Module):
    def __init__(self, args):
        super(SimpleMatcher, self).__init__()

        one_step = (args.scale_max - args.scale_min) / 5
        scales = torch.arange(
            start=args.scale_min,
            end=args.scale_max + one_step + 0.01,
            step=one_step,
            dtype=torch.float32,
            device=args.device,
        )
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self._default_boxes = to_device(
            gen_default_boxes(
                shapes=[37, 19, 10, 5, 3, 1],
                scales=scales,
                aspect_ratios=aspect_ratios,
            ),
            args.device,
        )
        self._device = args.device

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str : list[torch.Tensor]],
        targets: dict[str : list[torch.Tensor]],
    ):
        all_boxes, gt_boxes = outputs["boxes"], targets["boxes"]

        gt_boxes_xyxy = []
        for b_gt in gt_boxes:
            gt_boxes_xyxy.append(from_cxcyhw_to_xyxy(b_gt))

        pred_boxes = []
        for def_boxes, boxes in zip(self._default_boxes, all_boxes):
            boxes_coord = torch.stack(
                [
                    def_boxes[..., 0] + def_boxes[..., 3] * boxes[..., 0],
                    def_boxes[..., 1] + def_boxes[..., 2] * boxes[..., 1],
                    def_boxes[..., 2] * boxes[..., 2].exp(),
                    def_boxes[..., 3] * boxes[..., 3].exp(),
                ],
                -1,
            ).flatten(1, 3)

            pred_boxes.append(boxes_coord)
        boxes_xyxy = from_cxcyhw_to_xyxy(torch.cat(pred_boxes, 1).contiguous())

        pairs, pos_inds, neg_inds = [], [], []
        for b_boxes, b_gt_boxes in zip(boxes_xyxy, gt_boxes_xyxy):
            ious = get_iou(bbox1=b_boxes, bbox2=b_gt_boxes)

            max_ind = torch.argmax(ious, 0)
            for gt_ind, pred_ind in enumerate(max_ind):
                ious[pred_ind, gt_ind] = 0.0

            b_pairs = torch.cat(
                [
                    torch.stack(
                        [
                            max_ind,
                            torch.arange(0, b_gt_boxes.shape[0], device=self._device),
                        ],
                        -1,
                    ),
                    torch.nonzero(ious >= 0.5, as_tuple=False),
                ],
                0,
            )
            pairs.append(b_pairs)

            neg_mask = torch.ones(
                size=(b_boxes.shape[0],), dtype=torch.bool, device=self._device
            )
            pos_ind = torch.unique(b_pairs[..., 0])

            neg_mask[pos_ind] = False
            neg_ind = torch.nonzero(neg_mask).flatten()

            pos_inds.append(pos_ind)
            neg_inds.append(neg_ind)

        return pairs, pos_inds, neg_inds


def build_matcher(matcher_cls, args):
    return matcher_cls(args)
