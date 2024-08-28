import pdb

import torch
from torch.nn import Module

from ...utils.bbox_utils import from_cxcyhw_to_xyxy


class PairSelfAttention(Module):
    def __init__(
        self, input_shape: tuple[int, int, int], output_shape: tuple[int, int, int]
    ) -> None:
        super(PairSelfAttention, self).__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        top_k_centers: torch.Tensor,
    ):
        batch_size = query.size(0)

        pairs = _get_pairs(top_k_centers)


def _get_pairs(top_k_centers: torch.Tensor):

    batch_size = top_k_centers.size(dim=0)
    num_objects = top_k_centers.size(dim=1)

    bbox_coord = from_cxcyhw_to_xyxy(top_k_centers)

    bbox_coord1 = bbox_coord.unsqueeze(dim=2)
    bbox_coord2 = bbox_coord.unsqueeze(dim=1)

    inter_mins = torch.maximum(bbox_coord1[..., :2], bbox_coord2[..., :2])
    inter_maxs = torch.maximum(bbox_coord1[..., 2:], bbox_coord2[..., 2:])
    inter_wh = torch.clip(inter_maxs - inter_mins, min=0)

    inter_area = torch.mul(inter_wh[..., 0], inter_wh[..., 1])

    bbox_area = torch.mul(
        bbox_coord[..., 2] - bbox_coord[..., 0], bbox_coord[..., 3] - bbox_coord[..., 1]
    )
    bbox_area1 = bbox_area.unsqueeze(dim=-1)
    bbox_area2 = bbox_area.unsqueeze(dim=-2)

    bbox_union_area = bbox_area1 + bbox_area2 - inter_area
    # the IoU between two same objects is 1, we do not want that kind of pairs
    bbox_iou = inter_area / bbox_union_area - torch.eye(n=inter_area.size(dim=-1))

    # turn the indices from [[0, 2, 1]] to [[[0, 0], [1, 2], [2, 1]]]
    pair_idx = torch.stack(
        [
            torch.arange(start=0, end=num_objects)
            .unsqueeze(0)
            .broadcast_to(size=(batch_size, num_objects)),
            torch.argmax(bbox_iou, dim=-1),
        ],
        dim=-1,
    )

    # compute the L1-distance for each bbox
    bbox_l1 = torch.abs(bbox_coord[..., 2] - bbox_coord[..., 0]) + torch.abs(
        bbox_coord[..., 3] - bbox_coord[..., 1]
    )

    # get the corresponding L1-distance of each bbox
    # for example, the index pair for 0 is [0, 1], then we have [L1-distance for bbox0, L1-distance for bbox1]
    bbox_l1_pair = torch.stack(
        [bbox_l1, torch.gather(bbox_l1, index=pair_idx[..., 1], dim=-1)],
        dim=-1,
    )

    # determine the order of each pair based on their L1-distance
    # the index of the bbox with larger L1-distance is the first one index in the pair
    correct_order = torch.where(
        (bbox_l1_pair[..., 0] >= bbox_l1_pair[..., 1]).unsqueeze(-1),
        pair_idx,
        pair_idx.flip(-1),
    )

    return correct_order


if __name__ == "__main__":
    centers = (
        torch.Tensor([[4, 8, 2, 2], [7, 4, 4, 4], [2, 10, 2, 8]]).unsqueeze(0) / 20
    )

    _get_pairs(centers)
