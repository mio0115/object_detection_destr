import math

import torch
from torch.nn import Module

from ...utils.bbox_utils import from_cxcyhw_to_xyxy


class PairSelfAttention(Module):
    def __init__(self, heads_num: int) -> None:
        super(PairSelfAttention, self).__init__()

        self._heads_num = heads_num

    @property
    def heads_num(self):
        return self._heads_num

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        top_k_centers: torch.Tensor,
    ):
        """Implmentation based on DESTR: Object Detection with Split Transformer
        Instead of self-attention, authors use pair self-attention.
        The steps of pair self-attention are as following:
        1. Pairs up those components of input feature map.
        2. Compute a2 score.
        3. Compute o2 score.
        """
        batch_size, _, seq_len, d_model = query.shape

        """ The following block is to find indices of pairs based on their IoU. 
        Component a is paired up with Component a' if their IoU is larger than other components
        To compute A2, we need to pair up indices of (a, b) and (a', b')
        """
        pairs = _get_pairs(top_k_centers)

        # flipped_mask
        tmp = torch.arange(0, seq_len)[None, :, None].expand(batch_size, seq_len, 2)
        flipped_mask = torch.where(pairs == tmp, False, True)

        left_query = torch.stack(
            tensors=[
                b_query.index_select(dim=1, index=b_idx)
                for b_query, b_idx in zip(query, pairs[..., 0])
            ],
            dim=0,
        )
        left_key = torch.stack(
            tensors=[
                b_key.index_select(dim=1, index=b_idx)
                for b_key, b_idx in zip(key, pairs[..., 0])
            ],
            dim=0,
        )
        left_value = torch.stack(
            tensors=[
                b_val.index_select(dim=1, index=b_idx)
                for b_val, b_idx in zip(value, pairs[..., 0])
            ],
            dim=0,
        )

        right_query = torch.stack(
            tensors=[
                b_query.index_select(dim=1, index=b_idx)
                for b_query, b_idx in zip(query, pairs[..., 1])
            ],
            dim=0,
        )
        right_key = torch.stack(
            tensors=[
                b_key.index_select(dim=1, index=b_idx)
                for b_key, b_idx in zip(key, pairs[..., 1])
            ],
            dim=0,
        )
        right_value = torch.stack(
            tensors=[
                b_val.index_select(dim=1, index=b_idx)
                for b_val, b_idx in zip(value, pairs[..., 1])
            ],
            dim=0,
        )
        value = torch.concat([left_value, right_value], dim=-1)

        """ We compute A2(a, b) = <q_{pi a}, k_{pi b}> + <q_{pi a'}, k_{pi b'}> """
        a2_l = torch.matmul(left_query, left_key.transpose(2, 3))
        a2_r = torch.matmul(right_query, right_key.transpose(2, 3))

        a2 = a2_l + a2_r

        a2 = a2.softmax(dim=-1) / math.sqrt(2 * d_model)
        o2 = torch.matmul(a2, value).transpose(1, 2).flatten(2)

        o2 = (
            o2.reshape(batch_size, seq_len, 2, d_model * self.heads_num)
            .masked_fill(mask=flipped_mask.unsqueeze(-1), value=float(0))
            .sum(2, keepdim=False)
        )

        return o2


def _get_pairs(top_k_centers: torch.Tensor):
    """According to DESTR, pair self-attention has better performance than self-attention.
    For each object query, we only take the pair which has the highest IoU.
    Then order the pair by their L1-distance decreasingly.
    """
    batch_size, num_objects, *_ = top_k_centers.shape

    bbox_coord = from_cxcyhw_to_xyxy(top_k_centers)

    bbox_coord1 = bbox_coord.unsqueeze(dim=2)
    bbox_coord2 = bbox_coord.unsqueeze(dim=1)

    inter_mins = torch.maximum(bbox_coord1[..., :2], bbox_coord2[..., :2])
    inter_maxs = torch.minimum(bbox_coord1[..., 2:], bbox_coord2[..., 2:])
    inter_wh = inter_maxs - inter_mins

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
