import torch


class NestedTensor(object):
    def __init__(self, tensor: torch.Tensor, mask) -> None:
        self.tensor = tensor
        self.mask = mask

    def to(self, new_device: torch.device):
        new_tensor = self.tensor.to(new_device)

        new_mask = None
        if self.mask is not None:
            new_mask = self.mask.to(new_device)

        return NestedTensor(new_tensor, new_mask)

    def decompose(self):
        return self.tensor, self.mask


def nested_tensor_from_tensor_list(tensor_list: list[torch.Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size

        batch_size, channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(size=batch_shape, dtype=dtype, device=device)
        mask = torch.ones(size=(batch_size, height, width), dtype=dtype, device=device)

        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")

    return NestedTensor(tensor=tensor, mask=mask)


def _max_by_axis(the_list):
    # type: (list[list[int]]) -> list[int]
    maxes = the_list[0]

    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    return maxes
