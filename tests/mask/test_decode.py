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
    mask = torch.zeros((h, w), dtype=torch.uint8)

    # @case(id="start_area")
    def case_start_area(self):
        min = 0
        max = 5
        mask_pt = self.mask.clone()
        mask_pt[min:max, min:max] = 1
        return (min, max, self.h, self.w, mask_pt)

    def case_center_area(self):
        min = 5
        max = 10
        mask_pt = self.mask.clone()
        mask_pt[min:max, min:max] = 1
        return (min, max, self.h, self.w, mask_pt)

    def case_end_area(self):
        min = 20
        max = 25
        mask_pt = self.mask.clone()
        mask_pt[min:max, min:max] = 1
        return (min, max, self.h, self.w, mask_pt)

    def case_full_area(self):
        min = 0
        max = 25
        mask_pt = self.mask.clone()
        mask_pt[min:max, min:max] = 1
        return (min, max, self.h, self.w, mask_pt)


@pytest.mark.benchmark(group="decode", warmup=True)
@parametrize_with_cases("min, max, h, w, result", cases=DecodeCases)
def test_decode_pt(benchmark, min: int, max: int, h: int, w: int, result: Tensor):  # noqa: N802
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    # decode the mask
    rle_pt = tmask.encode(mask_pt)
    result_pt: Tensor = benchmark(tmask.decode, rle_pt)
    # compare the results
    assert torch.equal(result_pt, result)


@pytest.mark.benchmark(group="decode", warmup=True)
@parametrize_with_cases("min, max, h, w, result", cases=DecodeCases)
def test_decode_np(benchmark, min: int, max: int, h: int, w: int, result: Tensor):  # noqa: N802
    # create a mask
    mask_np = np.zeros((h, w), dtype=np.uint8, order="F")
    mask_np[min:max, min:max] = 1
    # decode the mask
    rle_np = mask.encode(mask_np)
    result_np = benchmark(mask.decode, rle_np)
    # compare the results
    assert np.array_equal(result_np, result.numpy())


@parametrize_with_cases("min, max, h, w, result", cases=DecodeCases)
def test_decode(min: int, max: int, h: int, w: int, result: Tensor):  # noqa: N802
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    mask_np = np.asfortranarray(mask_pt.numpy())
    # decode the mask
    rle_np = mask.encode(mask_np)
    rle_pt = tmask.encode(mask_pt)
    result_np = mask.decode(rle_np)
    result_pt = tmask.decode(rle_pt)
    # compare the results
    # compare the results
    assert np.array_equal(result_np, mask_np)
    assert np.array_equal(result_np, mask_pt.numpy())
    assert np.array_equal(result_np, result_pt.numpy())
