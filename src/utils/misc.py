import torch
import numpy as np


class NestedTensor(object):
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, new_device: torch.device):
        new_tensors = self.tensors.to(new_device)

        new_mask = None
        if self.mask is not None:
            new_mask = self.mask.to(new_device)

        return NestedTensor(new_tensors, new_mask)

    def decompose(self):
        return self.tensors, self.mask


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

    return NestedTensor(tensors=tensor, mask=mask)


def _max_by_axis(the_list):
    # type: (list[list[int]]) -> list[int]
    maxes = the_list[0]

    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    return maxes


def inverse_sigmoid(tensor, epsilon: float = 1e-7):
    assert epsilon > 0, "epsilon must large than 0"

    return -1 * (tensor.clip(min=epsilon).pow(-1) - 1).log()


def to_device(inputs, device):
    if isinstance(inputs, (torch.Tensor, torch.nn.Module)):
        new_inputs = inputs.to(device=device)
    elif isinstance(inputs, (tuple, list)):
        new_inputs = [to_device(inp, device=device) for inp in inputs]
    elif isinstance(inputs, dict):
        new_inputs = {}

        for key, item in inputs.items():
            new_inputs[key] = to_device(item, device=device)
    else:
        raise ValueError(
            f"inputs should be one of following types: torch.Tensor, tuple, list or dict. But got {type(inputs)}"
        )

    return new_inputs

def reduce_dict(dict_, weights, default_weight: float = 1.0):
    sum_ = 0
    
    for key, val in dict_.items():
        sum_ += val * weights.get(key, default_weight)
    
    return sum_

def np_softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis=axis, keepdims=True))
    f_x = y / np.sum(y, axis=axis, keepdims=True)

    return f_x
