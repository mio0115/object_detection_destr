import math
from enum import Enum
from typing import Iterable

import torch
import numpy as np

from .misc import make_grid


class BBoxType(Enum):
    CXCYHW = "cxcyhw"
    XYXY = "xyxy"
    XYHW = "xyhw"


def from_np_to_tensor(func):
    def helper(*args, **kwargs):
        if "bbox1" in kwargs.keys() and isinstance(kwargs["bbox1"], np.ndarray):
            kwargs["bbox1"] = torch.from_numpy(kwargs["bbox1"])
        if "bbox2" in kwargs.keys() and isinstance(kwargs["bbox2"], np.ndarray):
            kwargs["bbox2"] = torch.from_numpy(kwargs["bbox2"])
        if "bbox_coord" in kwargs.keys() and isinstance(
            kwargs["bbox_coord"], np.ndarray
        ):
            kwargs["bbox_coord"] = torch.from_numpy(kwargs["bbox_coord"])

        return func(*args, **kwargs)

    return helper


@from_np_to_tensor
def from_cxcyhw_to_xyxy(
    bbox_coord: torch.Tensor, min_val: float = 0, max_val: float = 1
) -> torch.Tensor:
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
    if bbox_coord.shape[0] == 0:
        return bbox_coord

    new_bbox_coord = torch.stack(
        [
            torch.clip(bbox_coord[..., 0] - bbox_coord[..., 3] / 2, min=min_val),
            torch.clip(bbox_coord[..., 1] - bbox_coord[..., 2] / 2, min=min_val),
            torch.clip(bbox_coord[..., 0] + bbox_coord[..., 3] / 2, max=max_val),
            torch.clip(bbox_coord[..., 1] + bbox_coord[..., 2] / 2, max=max_val),
        ],
        dim=-1,
    )

    return new_bbox_coord


@from_np_to_tensor
def from_xyxy_to_cxcyhw(
    bbox_coord: torch.Tensor, min_val: float = 0, max_val: float = 1
) -> torch.Tensor:
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
    if bbox_coord.shape[0] == 0:
        return bbox_coord

    new_bbox_coord = torch.stack(
        [
            torch.clip(
                (bbox_coord[..., 0] + bbox_coord[..., 2]) / 2, min=min_val, max=max_val
            ),
            torch.clip(
                (bbox_coord[..., 1] + bbox_coord[..., 3]) / 2, min=min_val, max=max_val
            ),
            torch.clip(
                bbox_coord[..., 3] - bbox_coord[..., 1], min=min_val, max=max_val
            ),
            torch.clip(
                bbox_coord[..., 2] - bbox_coord[..., 0], min=min_val, max=max_val
            ),
        ],
        dim=-1,
    )

    return new_bbox_coord


@from_np_to_tensor
def from_xywh_to_xyxy(
    bbox_coord: torch.Tensor, min_val: float = 0, max_val: float = 1
) -> torch.Tensor:
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
    if bbox_coord.shape[0] == 0:
        return bbox_coord

    new_bbox_coord = torch.concat(
        [
            bbox_coord[..., :2],
            torch.stack(
                [
                    torch.clip(bbox_coord[..., 0] + bbox_coord[..., 2], max=max_val),
                    torch.clip(bbox_coord[..., 1] + bbox_coord[..., 3], max=max_val),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )

    return new_bbox_coord


def check_bbox_xyxy(bbox_coord):
    if bbox_coord.min() < 0:
        print(f"Warning: negative coord: {bbox_coord}")
    if (bbox_coord[..., 0] >= bbox_coord[..., 2]).any():
        print(f"Warning: non-positive width {bbox_coord}")
    if (bbox_coord[..., 1] >= bbox_coord[..., 3]).any():
        print(f"Warning: non-positive height: {bbox_coord}")


def check_bbox_cxcyhw(bbox_coord):
    if bbox_coord[..., 2].min() <= 0:
        print(f"Warning: non-positive height {bbox_coord}")
    if bbox_coord[..., 3].min() <= 0:
        print(f"Warning: non-positive width {bbox_coord}")
    if bbox_coord[..., :2].min() < 0:
        print(f"Warning: negative center {bbox_coord}")


def complete_iou(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, epsilon=1e-6):
    pred_cxcyhw = from_xyxy_to_cxcyhw(pred_xyxy)
    gt_cxcyhw = from_xyxy_to_cxcyhw(gt_xyxy)

    iou = get_iou(pred_xyxy, gt_xyxy)

    # compute diagonal length of minimal boxes containing predicted bbox and corresponding ground truth bbox
    minimal_box_wh = torch.maximum(
        pred_xyxy[:, None, 2:], gt_xyxy[None, :, 2:]
    ) - torch.minimum(pred_xyxy[:, None, :2], gt_xyxy[None, :, :2])
    minimal_box_wh = minimal_box_wh.clamp(min=0)
    diag_len = minimal_box_wh.pow(2).sum(-1)

    # compute distance between centers of predicted bbox and corresponding ground truth bbox
    center_wh = torch.abs(pred_cxcyhw[:, None, :2] - gt_cxcyhw[None, :, :2])
    center_dist = center_wh.pow(2).sum(-1)

    # compute V and alpha
    v = (
        4
        / pow(torch.pi, 2)
        * torch.pow(
            torch.atan(gt_cxcyhw[..., 3] / gt_cxcyhw[..., 2].clamp(min=epsilon))[
                None, :
            ]
            - torch.atan(pred_cxcyhw[..., 3] / pred_cxcyhw[..., 2].clamp(min=epsilon))[
                :, None
            ],
            2,
        )
    )
    with torch.no_grad():
        large_iou = (iou > 0.5).float()
        alpha = large_iou * (v / (1 - iou + v))

    cious = iou - center_dist / diag_len.clamp(min=epsilon) - alpha * v
    cious = cious.clamp(min=-1.0, max=1.0)

    return 1 - cious


@from_np_to_tensor
def get_iou(bbox1, bbox2, epsilon=1e-6):

    inter_mins = torch.maximum(bbox1[:, None, :2], bbox2[None, :, :2])
    inter_maxs = torch.minimum(bbox1[:, None, 2:], bbox2[None, :, 2:])
    inter_wh = torch.clamp(inter_maxs - inter_mins, min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    bbox1_area = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    bbox2_area = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    union_area = bbox1_area[:, None] + bbox2_area[None, :] - inter_area

    iou = inter_area / union_area.clamp(epsilon)

    return iou


def filter_flat_box(boxes, epsilon=1e-6):
    """Filter boxes with height or weight is 0"""

    boxes_wh = torch.stack(
        [boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]], dim=1
    )
    boxes_zero = (boxes_wh > epsilon).all(-1)

    filtered_boxes = boxes[boxes_zero]

    return filtered_boxes


def gen_default_boxes(
    shapes: Iterable[int],
    scales: Iterable[float],
    aspect_ratios: Iterable[float],
):
    default_boxes = []

    for ind, (shape, ar) in enumerate(zip(shapes, aspect_ratios)):
        num_boxes = (len(ar) + 1) * 2
        centers = (
            make_grid(height=shape, width=shape, bias=0.5, norm=True)
            .unsqueeze(2)
            .repeat(1, 1, num_boxes, 1)
        )

        scale = scales[ind]
        g_scale = math.sqrt(scales[ind] * scales[ind + 1])

        hw_pairs = [(scale, scale), (g_scale, g_scale)]
        for ar_ in ar:
            sqrt_ar = math.sqrt(ar_)
            hw_pairs.append((scale * sqrt_ar, scale / sqrt_ar))
            hw_pairs.append((scale / sqrt_ar, scale * sqrt_ar))
        hw_pairs = torch.tensor(hw_pairs)[None, None, :, :].repeat(shape, shape, 1, 1)

        # shape of elements in default_boxes is (bs, h, w, boxes, 4)
        default_boxes.append(torch.cat([centers, hw_pairs], -1).unsqueeze(0))

    return default_boxes


def update_coord_new_boundary(
    boxes,
    new_boundary: tuple[int, int, int, int],
    origin_boundary: tuple[int, int],
):
    if boxes.shape[0] == 0:
        return boxes
    boxes_xyxy = torch.stack(
        [
            torch.clip(boxes[..., 0] - boxes[..., 3] / 2, min=0),
            torch.clip(boxes[..., 1] - boxes[..., 2] / 2, min=0),
            torch.clip(boxes[..., 0] + boxes[..., 3] / 2, max=origin_boundary[0]),
            torch.clip(boxes[..., 1] + boxes[..., 2] / 2, max=origin_boundary[1]),
        ],
        dim=-1,
    )

    new_boxes = torch.stack(
        [
            boxes_xyxy[..., 0].clamp(max=new_boundary[0]),
            boxes_xyxy[..., 1].clamp(max=new_boundary[1]),
            boxes_xyxy[..., 2].clamp(min=new_boundary[2]),
            boxes_xyxy[..., 3].clamp(min=new_boundary[3]),
        ],
        -1,
    )

    boxes_cxcyhw = torch.stack(
        [
            torch.clip(
                (new_boxes[..., 0] + new_boxes[..., 2]) / 2,
                min=0,
                max=origin_boundary[0],
            ),
            torch.clip(
                (new_boxes[..., 1] + new_boxes[..., 3]) / 2,
                min=0,
                max=origin_boundary[1],
            ),
            torch.clip(
                new_boxes[..., 3] - new_boxes[..., 1], min=0, max=origin_boundary[0]
            ),
            torch.clip(
                new_boxes[..., 2] - new_boxes[..., 0], min=0, max=origin_boundary[1]
            ),
        ],
        dim=-1,
    )
    return boxes_cxcyhw


if __name__ == "__main__":
    bbox1 = torch.Tensor([[10, 50, 50, 100]])
    bbox2 = torch.Tensor([[20, 100, 100, 200]])

    complete_iou(bbox1, bbox2)
