import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_cases import case, parametrize_with_cases
import pytorchcocotools.mask as tmask
import torch
from torch import Tensor


class DecodeCases:
    h = 25
    w = 25

    # @case(id="start_area")
    def case_start_area(self):
        return (0, 5, self.h, self.w, {"size": [self.h, self.w], "counts": "5O5O"})

    def case_center_area(self):
        return (5, 10, self.h, self.w, {"size": [self.h, self.w], "counts": "5O5O"})

    def case_end_area(self):
        return (20, 25, self.h, self.w, {"size": [self.h, self.w], "counts": "5O5O"})

    def case_full_area(self):
        return (0, 25, self.h, self.w, {"size": [self.h, self.w], "counts": "5O5O"})


@pytest.mark.benchmark(group="decode", warmup=True)
@parametrize_with_cases("min, max, h, w, rle", cases=DecodeCases)
def test_encode_pt(benchmark, min: int, max: int, h: int, w: int, rle: dict):  # noqa: N802
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    # decode the mask
    rle_pt = tmask.encode(mask_pt)
    result_pt: Tensor = benchmark(tmask.decode, rle_pt)
    # compare the results
    assert result_pt == rle


@pytest.mark.benchmark(group="decode", warmup=True)
@parametrize_with_cases("min, max, h, w, rle", cases=DecodeCases)
def test_encode_np(benchmark, min: int, max: int, h: int, w: int, rle: dict):  # noqa: N802
    # create a mask
    mask_np = np.zeros((h, w), dtype=np.uint8, order="F")
    mask_np[min:max, min:max] = 1
    # decode the mask
    rle_np = mask.encode(mask_np)
    result_np = benchmark(mask.decode, rle_np)
    # compare the results
    assert result_np == rle


@parametrize_with_cases("min, max, h, w, rle", cases=DecodeCases)
def test_encode(min: int, max: int, h: int, w: int, rle: dict):  # noqa: N802
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    mask_np = np.asfortranarray(mask_pt.numpy())
    # decode the mask
    rle_np = mask.encode(mask_np)
    rle_pt = tmask.encode(mask_pt)
    result_np = mask.encode(rle_np)
    result_pt = tmask.encode(rle_pt)
    # compare the results
    # compare the results
    assert np.array_equal(result_np, mask_np)
    assert torch.equal(result_np, mask_pt)
    assert np.array_equal(result_np, result_pt.numpy())
