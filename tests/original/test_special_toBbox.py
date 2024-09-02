import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RleObj, RleObjs
import pytorchcocotools.mask as mask_util


def _encode(x: Tensor) -> RleObj | RleObjs:
    return mask_util.encode(x.to(dtype=torch.uint8))


def test_tobbox_full_image():
    mask = torch.tensor([[0, 1], [1, 1]])
    bbox = mask_util.toBbox(_encode(mask))
    assert torch.equal(bbox, torch.tensor([0, 0, 2, 2], dtype=torch.float32))


# bugfix by piotr in ff4a47150bf66
def test_tobbox_non_full_image():
    mask = torch.zeros((10, 10), dtype=torch.uint8)
    mask[2:4, 3:6] = 1
    bbox = mask_util.toBbox(_encode(mask))
    assert torch.equal(bbox, torch.tensor([3, 2, 3, 2], dtype=torch.float32))
