import torch
from torch import nn

from .bbox_utils import complete_iou


class SetCriterion(nn.Moduel):
    def __init__(self, num_classes, matcher, loss_weight, loss_fn):
        super(SetCriterion, self).__init__()

        self._num_cls = num_classes
        self._matcher = matcher
        self._loss_weights = loss_weight
        self._loss_fns = loss_fn

    def _reduce_dict(self, losses):
        total_loss = 0

        for key, weight in self._loss_weights.item():
            total_loss += weight * losses.get(key, 0).sum(-1)

        return total_loss

    def forward(self, outputs, targets):
        losses = {}

        indices = self._matcher(outputs, targets)

        # In order to compute loss, we reorder the targets
        gt_bboxes = torch.stack(
            [
                b_tgt.index_select(index=b_idx, dim=0)
                for b_idx, b_tgt in zip(indices, targets[0])
            ],
            dim=0,
        )
        gt_class = torch.stack(
            [
                b_tgt.index_select(index=b_idx, dim=0)
                for b_idx, b_tgt in zip(indices, targets[1])
            ],
            dim=0,
        )

        # loss of class
        losses["class"] = self._loss_fns["class"](outputs["pred_class"], gt_class)

        # loss of boundary box
        losses["bbox"] = self._loss_fns["bbox"](outputs["pred_bbox"], gt_bboxes)

        # loss of complete IoU
        losses["ciou"] = self._loss_fns["ciou"](outputs["pred_bbox"], gt_bboxes)

        return self._reduce_dict(losses)


class CompleteIOULoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, outputs, gt):
        ciou = complete_iou(outputs, gt)

        return ciou.mean()
