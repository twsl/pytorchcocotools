import pytest
from pytorchcocotools import _maskApi
import torch
from torch import Tensor


@pytest.fixture
def mask() -> Tensor:
    min, max = 10, 20
    pt_mask = torch.zeros((25, 25, 2), dtype=torch.uint8)
    pt_mask[min:max, min:max, :] = 1
    return pt_mask


def test_rleEncode(mask: Tensor):  # noqa: N802
    rle = _maskApi.rleEncode(mask, mask.shape[0], mask.shape[1], mask.shape[2])
    # check the result
    assert rle
