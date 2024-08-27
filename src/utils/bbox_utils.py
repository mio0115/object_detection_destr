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
            torch.maximum(bbox_coord[..., 0] - bbox_coord[..., 3] / 2, 0),
            torch.maximum(bbox_coord[..., 1] - bbox_coord[..., 2] / 2, 0),
            torch.minimum(bbox_coord[..., 0] + bbox_coord[..., 3] / 2, 1),
            torch.minimum(bbox_coord[..., 1] + bbox_coord[..., 2] / 2, 1),
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
