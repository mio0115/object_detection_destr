import torch
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

    def _get_ciou_loss(self, pred_boxes, gt_boxes):
        return self._loss_fns["ciou"](pred_boxes, gt_boxes)

    def _get_bbox_loss(self, pred_boxes, gt_boxes):
        return self._loss_fns["bbox"](pred_boxes, gt_boxes)

    def _get_class_loss(self, pred_logits, pred_idx, gt_class):
        selected_objects = pred_logits.index_select(index=pred_idx, dim=0)

        mask = torch.ones(
            pred_logits.size(0), dtype=torch.bool, device=pred_logits.device
        )
        mask[pred_idx] = False

        remaining_objects = pred_logits[mask]
        ordered_logits = torch.concat([selected_objects, remaining_objects], dim=0)

        objects_num = pred_logits.size(0)
        dummy_class = torch.tensor([[0, 1]], device=pred_logits.device).expand(
            (objects_num - pred_idx.size(0), 2)
        )
        gt_class = torch.concat([gt_class, dummy_class], dim=0)

        return self._loss_fns["class"](ordered_logits, gt_class)

    def forward(self, outputs, targets):
        losses = {"class": 0, "bbox": 0, "ciou": 0}

        indices = to_device(
            self._matcher(outputs, targets), device=outputs["pred_class"].device
        )

        for b_output_cls, b_output_box, b_targets, b_idx in zip(
            outputs["pred_class"], outputs["pred_boxes"], targets, indices
        ):
            output_pred_boxes = b_output_box.index_select(index=b_idx[0], dim=0)
            gt_class = b_targets["labels"].index_select(index=b_idx[1], dim=0)
            gt_boxes = b_targets["boxes"].index_select(index=b_idx[1], dim=0)

            losses["class"] += self._get_class_loss(b_output_cls, b_idx[0], gt_class)
            if b_idx[0].size(0) > 0:
                losses["bbox"] += self._get_bbox_loss(output_pred_boxes, gt_boxes)
                losses["ciou"] += self._get_ciou_loss(output_pred_boxes, gt_boxes)

        return self._reduce_dict(losses, len(indices))


class CompleteIOULoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, outputs, gt):
        ciou = complete_iou(outputs, gt)

        return ciou.mean()
