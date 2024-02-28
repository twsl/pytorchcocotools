import torch
from torch import Tensor


class BaseCases:
    h = 25
    w = 25
    mask = torch.zeros((h, w), dtype=torch.uint8)

    def _build_mask(self, min1: int, max1: int) -> Tensor:
        mask_pt1 = self.mask.clone()
        mask_pt1[min1:max1, min1:max1] = 1
        return mask_pt1
