from torch import nn

from .bbox_utils import complete_iou
from ..utils.misc import to_device


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, loss_weight, loss_fn):
        super(SetCriterion, self).__init__()

        self._num_cls = num_classes
        self._matcher = matcher
        self._loss_weights = loss_weight
        self._loss_fns = loss_fn

    def _reduce_dict(self, losses, batch_size):
        total_loss = 0

        for key, weight in self._loss_weights.items():
            total_loss += weight * losses.get(key, 0).sum(-1)

        return total_loss / batch_size

    def forward(self, outputs, targets):
        losses = {"class": 0, "bbox": 0, "ciou": 0}

        indices = to_device(
            self._matcher(outputs, targets), device=outputs["pred_class"].device
        )

        for b_output_cls, b_output_box, b_targets, b_idx in zip(
            outputs["pred_class"], outputs["pred_boxes"], targets, indices
        ):
            output_pred_class = b_output_cls.index_select(index=b_idx[0], dim=0)
            output_pred_boxes = b_output_box.index_select(index=b_idx[0], dim=0)
            gt_class = b_targets["labels"].index_select(index=b_idx[1], dim=0)
            gt_boxes = b_targets["boxes"].index_select(index=b_idx[1], dim=0)

            losses["class"] += self._loss_fns["class"](output_pred_class, gt_class)
            losses["bbox"] += self._loss_fns["bbox"](output_pred_boxes, gt_boxes)
            losses["ciou"] += self._loss_fns["ciou"](output_pred_boxes, gt_boxes)

        return self._reduce_dict(losses, len(indices))


class CompleteIOULoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, outputs, gt):
        ciou = complete_iou(outputs, gt)

        return ciou.mean()
