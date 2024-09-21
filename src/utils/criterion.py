import torch
from torch import nn
import numpy as np

from .bbox_utils import (
    complete_iou,
    get_iou,
    from_xyxy_to_cxcyhw,
    from_cxcyhw_to_xyxy,
    gen_default_boxes,
)
from ..utils.misc import to_device, np_softmax


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, loss_fn):
        super(SetCriterion, self).__init__()

        self._num_cls = num_classes
        self._matcher = matcher
        self._loss_fns = loss_fn

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
        dummy_class = torch.ones(
            (objects_num - pred_idx.size(0),), device=pred_logits.device
        ).long()
        gt_class = torch.concat([gt_class, dummy_class], dim=0)
        oh_gt_class = nn.functional.one_hot(gt_class, num_classes=2)

        return self._loss_fns["class"](
            ordered_logits, oh_gt_class, ordered_logits.size(0)
        )

    def forward(self, outputs, targets):
        losses = {"class": [], "bbox": [], "ciou": []}
        device = outputs["pred_class"].device

        indices = to_device(self._matcher(outputs, targets), device=device)

        for b_output_cls, b_output_box, b_targets, b_idx in zip(
            outputs["pred_class"], outputs["pred_boxes"], targets, indices
        ):
            output_pred_boxes = from_cxcyhw_to_xyxy(b_output_box).index_select(
                index=b_idx[0], dim=0
            )
            gt_class = b_targets["labels"].index_select(index=b_idx[1], dim=0)
            gt_boxes = b_targets["boxes"].index_select(index=b_idx[1], dim=0)

            losses["class"].append(
                self._get_class_loss(b_output_cls, b_idx[0], gt_class)
            )
            if b_idx[0].size(0) > 0:
                losses["bbox"].append(self._get_bbox_loss(output_pred_boxes, gt_boxes))
                losses["ciou"].append(self._get_ciou_loss(output_pred_boxes, gt_boxes))

        # average the batch
        for key, val in losses.items():
            if val:
                losses[key] = torch.stack(val).mean().to(device=device)
            else:
                losses[key] = torch.zeros(1, device=device)
        return losses


class CompleteIOULoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, outputs, gt):
        ciou = complete_iou(outputs, gt)

        return ciou.mean()


class MeanAveragePrecision(nn.Module):
    # consider the case IoU >= 0.5 first

    def __init__(self, num_cls: int = 1, threshold: float = 0.5, num_pred: int = 300):
        super(MeanAveragePrecision, self).__init__()

        self._num_cls = num_cls
        self._num_gts = 0
        self._num_pred = num_pred
        self._true_positives = np.zeros(num_pred)
        self._false_positives = np.zeros(num_pred)
        self._threshold = threshold

    def _compute_precision_recall(self):
        cumsum_tp = np.cumsum(self._true_positives)
        cumsum_fp = np.cumsum(self._false_positives)

        # num_gt = true positives + false negatives eq. every ground truth
        recall = cumsum_tp / self._num_gts
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)

        return precision, recall

    def _compute_ap(self, precision, recall):
        ap = 0.0

        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0

        return ap

    def reset(self):
        self._num_gts = 0
        self._true_positives = np.zeros(self._num_pred)
        self._false_positives = np.zeros(self._num_pred)

    @torch.no_grad()
    def compute(self):
        p, r = self._compute_precision_recall()

        # TODO: upgrade to multi-class
        ap = self._compute_ap(precision=p, recall=r)

        return ap

    @torch.no_grad()
    def forward(self, outputs, targets):
        outputs, targets = to_device(outputs, "cpu"), to_device(targets, "cpu")

        for cls_ in range(self._num_cls):

            for b_pr_logits, b_pr_boxes, b_gt in zip(
                outputs["pred_class"].numpy(),
                outputs["pred_boxes"].numpy(),
                targets,
            ):
                b_gt_class = nn.functional.one_hot(
                    b_gt["labels"], num_classes=self._num_cls + 1
                ).numpy()
                b_gt_boxes = b_gt["boxes"].numpy()
                b_pr_boxes = from_cxcyhw_to_xyxy(bbox_coord=b_pr_boxes).numpy()

                b_pr_prob = np_softmax(b_pr_logits, -1)
                b_pr_class = b_pr_prob.argmax(-1)

                cls_pr_idx = np.where(b_pr_class == cls_)[0]
                cls_gt_idx = np.where(b_gt_class == cls_)[0]

                if len(cls_gt_idx) == 0:
                    continue

                cls_pr_boxes = b_pr_boxes[cls_pr_idx]
                cls_pr_prob = b_pr_prob[cls_pr_idx]
                cls_gt_boxes = b_gt_boxes[cls_gt_idx]

                sorted_idx = np.argsort(-cls_pr_prob, axis=0)[:, cls_]
                cls_pr_boxes = cls_pr_boxes[sorted_idx]
                cls_pr_prob = cls_pr_prob[sorted_idx]

                num_gt = len(cls_gt_boxes)
                self._num_gts += num_gt

                matched_gt_boxes = np.zeros(num_gt)

                ious = get_iou(bbox1=cls_pr_boxes, bbox2=cls_gt_boxes).numpy()

                for i, iou in enumerate(ious):
                    best_iou_idx = np.argmax(iou)
                    best_iou = iou[best_iou_idx]

                    if (
                        best_iou >= self._threshold
                        and matched_gt_boxes[best_iou_idx] == 0
                    ):
                        self._true_positives[i] += 1
                        matched_gt_boxes[best_iou_idx] = 1
                    else:
                        self._false_positives[i] += 1


class SSDCriterion(nn.Module):
    def __init__(
        self,
        matcher: nn.Module,
        loss_fns: dict[str : nn.Module],
        loss_coef: float,
        *args,
        **kwargs,
    ):
        super(SSDCriterion, self).__init__(*args, **kwargs)

        self._matcher = matcher
        self._loss_fns = loss_fns
        self._loss_coef = loss_coef

    def forward(
        self,
        outputs: dict[str : list[torch.Tensor]],
        targets: dict[str : list[torch.Tensor]],
    ):
        pairs, pos_inds, neg_inds = self._matcher(outputs, targets)

        local_loss = self._loss_fns["local"](
            outputs["boxes"], targets["boxes"], pairs
        ).mean()
        class_loss = self._loss_fns["class"](
            outputs["conf"], targets["labels"], pairs, pos_inds, neg_inds
        ).mean()

        return self._loss_coef * class_loss + (1 - self._loss_coef) * local_loss


class SSDLocalCriterion(nn.Module):
    def __init__(self, args):
        super(SSDLocalCriterion, self).__init__()

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
        # new shape of default_boxes: (bs, seq_len, 4)
        self._default_boxes = torch.cat(
            [dboxes.flatten(1, 3) for dboxes in self._default_boxes], 1
        ).squeeze(0)

    # ruff
    def forward(
        self,
        boxes: list[torch.Tensor],
        gt_boxes: list[torch.Tensor],
        pairs: list[torch.Tensor],
    ):
        # format of boxes: cxcyhw
        # format of gt_boxes: xyxy
        # pairs: (pred_ind, gt_ind)

        flatten_boxes = []
        for box_coords in boxes:
            flatten_boxes.append(box_coords.flatten(1, 3))
        flatten_boxes = torch.cat(flatten_boxes, 1)

        all_losses = []
        for b_pairs, b_boxes, b_gt_boxes in zip(pairs, flatten_boxes, gt_boxes):
            db_ind = b_pairs[:, 0]  # indices for default boxes
            gt_ind = b_pairs[:, 1]  # indices for ground-truth

            cx = (
                b_gt_boxes[gt_ind, 0] - self._default_boxes[db_ind, 0]
            ) / self._default_boxes[db_ind, 3]
            cy = (
                b_gt_boxes[gt_ind, 1] - self._default_boxes[db_ind, 1]
            ) / self._default_boxes[db_ind, 2]
            h = torch.log(b_gt_boxes[gt_ind, 2] / self._default_boxes[db_ind, 2])
            w = torch.log(b_gt_boxes[gt_ind, 3] / self._default_boxes[db_ind, 3])

            b_gt_boxes = torch.stack([cx, cy, h, w], -1)
            b_sel_boxes = b_boxes[db_ind]

            loss = nn.functional.smooth_l1_loss(b_sel_boxes, b_gt_boxes)

            all_losses.append(loss)

        return torch.stack(all_losses).mean()


class SSDClassCriterion(nn.Module):
    def __init__(self):
        super(SSDClassCriterion, self).__init__()

    def forward(
        self,
        logits: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        pairs: list[torch.Tensor],
        pos_inds: list[torch.Tensor],
        neg_inds: list[torch.Tensor],
    ):

        flatten_logits = []
        for logit in logits:
            flatten_logits.append(logit.flatten(1, 3))
        flatten_logits = torch.cat(flatten_logits, 1)

        all_losses = []
        for b_logits, b_gt_labels, b_pairs, b_pos_ind, b_neg_ind in zip(
            flatten_logits, gt_labels, pairs, pos_inds, neg_inds
        ):

            ind = b_pairs[:, 0]
            gt_ind = b_pairs[:, 1]

            # reorder ground-truth labels for positive objects
            b_gt_labels = b_gt_labels[gt_ind]

            # compute confidence score for all objects
            b_conf = b_logits.softmax(-1)
            # select confidence score of POSITIVE objects
            b_pos_conf = b_conf[ind, b_gt_labels]
            # select confidence score of NEGATIVE objects
            b_neg_conf = b_conf[b_neg_ind, -1]
            # then apply hard negative mining to balance number between pos and neg
            preserved_neg = min(b_pos_ind.shape[0] * 3, b_neg_ind.shape[0])
            b_neg_conf, _ = torch.sort(b_neg_conf, dim=-1, descending=True)
            b_neg_conf = b_neg_conf[:preserved_neg]

        loss = (b_pos_conf.log().sum() + b_neg_conf.log().sum()) * -1
        all_losses.append(loss)

        return torch.stack(all_losses).mean()


if __name__ == "__main__":
    map_compute = MeanAveragePrecision()

    outputs = {
        "pred_class": torch.rand((2, 10, 2)),
        "pred_boxes": torch.rand((2, 10, 4)),
    }
    targets = (
        {"labels": torch.zeros(4), "boxes": torch.rand(4, 4)},
        {"labels": torch.zeros(8), "boxes": torch.rand(8, 4)},
    )

    val = map_compute(outputs, targets)
    print(val)
