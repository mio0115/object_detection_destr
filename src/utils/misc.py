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
