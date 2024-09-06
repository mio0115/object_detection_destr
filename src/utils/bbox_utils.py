import math

import torch


def from_cxcyhw_to_xyxy(bbox_coord: torch.Tensor) -> torch.Tensor:
    """
    Transform the bbox coordinates
    from (center_x, center_y, height, width)
    to
    (min_x, min_y, max_x, max_y)
    we make min_x and min_y >= 0

    Args:
        bbox_coord: Coordinates of boundary box. (cxcyhw)

    Returns:
        torch.Tensor: Coordinates of the boundary box. (xyxy)
    """

    new_bbox_coord = torch.stack(
        [
            torch.clip(bbox_coord[..., 0] - bbox_coord[..., 3] / 2, min=0),
            torch.clip(bbox_coord[..., 1] - bbox_coord[..., 2] / 2, min=0),
            torch.clip(bbox_coord[..., 0] + bbox_coord[..., 3] / 2, max=1),
            torch.clip(bbox_coord[..., 1] + bbox_coord[..., 2] / 2, max=1),
        ],
        dim=-1,
    )

    return new_bbox_coord


def from_xyxy_to_cxcyhw(bbox_coord: torch.Tensor) -> torch.Tensor:
    """
    Transform the bbox coordinates
    from (min_x, min_y, max_x, max_y)
    to
    (center_x, center_y, height, width)

    Args:
        bbox_coord: Coordinates of boundary box. (cxcyhw)

    Returns:
        torch.Tensor: Coordinates of the boundary box. (xyxy)
    """

    new_bbox_coord = torch.stack(
        [
            (bbox_coord[..., 0] + bbox_coord[..., 2]) / 2,
            (bbox_coord[..., 1] + bbox_coord[..., 3]) / 2,
            bbox_coord[..., 3] - bbox_coord[..., 1],
            bbox_coord[..., 2] - bbox_coord[..., 0],
        ],
        dim=-1,
    )

    return new_bbox_coord


def from_xywh_to_xyxy(bbox_coord: torch.Tensor) -> torch.Tensor:
    """
    Transform the bbox coordinates
    from (min_x, min_y, width, height)
    to
    (min_x, min_y, max_x, max_y)

    Args:
        bbox_coord: Coordinates of boundary box. (xyhw)

    Returns:
        torch.Tensor: Coordinates of the boundary box. (xyxy)
    """

    new_bbox_coord = torch.concat(
        [
            bbox_coord[..., :2],
            torch.stack(
                [
                    bbox_coord[..., 0] + bbox_coord[..., 2],
                    bbox_coord[..., 1] + bbox_coord[..., 3],
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )

    return new_bbox_coord


def complete_iou(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor):
    pred_cxcyhw = from_xyxy_to_cxcyhw(pred_xyxy)
    gt_cxcyhw = from_xyxy_to_cxcyhw(gt_xyxy)

    iou = get_iou(pred_xyxy, gt_xyxy)

    # breakpoint()

    # compute diagonal length of minimal boxes containing predicted bbox and corresponding ground truth bbox
    minimal_box_wh = torch.maximum(
        pred_xyxy[:, None, 2:], gt_xyxy[None, :, 2:]
    ) - torch.minimum(pred_xyxy[:, None, :2], gt_xyxy[None, :, :2])
    diag_len = minimal_box_wh.pow(2).sum(-1).sqrt()

    # compute distance between centers of predicted bbox and corresponding ground truth bbox
    center_wh = torch.abs(pred_cxcyhw[:, None, :2] - gt_cxcyhw[None, :, :2])
    center_dist = center_wh.pow(2).sum(-1).sqrt()

    # compute V and alpha
    v = (
        4
        / pow(math.pi, 2)
        * torch.pow(
            torch.atan(gt_cxcyhw[..., 3] / gt_cxcyhw[..., 2])[None, :]
            - torch.atan(pred_cxcyhw[..., 3] / pred_cxcyhw[..., 2])[:, None],
            2,
        )
    )
    alpha = torch.where(iou < 0.5, 0, v / (1 - iou + v))

    # if v.isnan().any() or alpha.isnan().any() or (center_dist == 0).any():
    #    breakpoint()

    return (1 - iou) + diag_len / center_dist + alpha * v


def get_iou(bbox1, bbox2):

    # breakpoint()

    inter_mins = torch.maximum(bbox1[:, None, :2], bbox2[None, :, :2])
    inter_maxs = torch.minimum(bbox1[:, None, 2:], bbox2[None, :, 2:])
    inter_wh = inter_maxs - inter_mins
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    bbox1_area = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    bbox2_area = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    bbox_area_sum = bbox1_area[:, None] + bbox2_area[None, :]

    iou = bbox_area_sum / inter_area

    return iou


def filter_flat_box(boxes, epsilon=1e-6):
    """Filter boxes with height or weight is 0"""

    # breakpoint()

    boxes_wh = torch.stack(
        [boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]], dim=1
    )
    boxes_zero = (boxes_wh > epsilon).all(-1)

    filtered_boxes = boxes[boxes_zero]

    # breakpoint()

    return filtered_boxes
