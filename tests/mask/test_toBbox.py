import numpy as np
import pycocotools.mask as nmask
import pytest
from pytest_cases import case, parametrize_with_cases
import pytorchcocotools.mask as tmask
import torch
from torch import Tensor

from .base_cases import BaseCases


class BboxCases(BaseCases):
    def case_start_area(self):
        return (self._build_mask(0, 5), [0, 0, 5, 5])

    def case_center_area(self):
        return (self._build_mask(5, 10), [5, 5, 5, 5])

    def case_end_area(self):
        return (self._build_mask(20, 25), [20, 20, 5, 5])

    def case_full_area(self):
        return (self._build_mask(0, 25), [0, 0, 25, 25])

    def case_rect_area(self):
        mask = torch.zeros((self.h, self.w + 10), dtype=torch.uint8)
        mask[10:25, 10:30] = 1
        return (mask, [10, 10, 20, 15])


@pytest.mark.benchmark(group="toBbox", warmup=True)
@parametrize_with_cases("mask, result", cases=BboxCases)
def test_toBbox_pt(benchmark, mask: Tensor, result: list[int]):  # noqa: N802
    # create a mask
    mask_pt = mask
    # compute the bounding box
    rle_pt = tmask.encode(mask_pt)
    result_pt: Tensor = benchmark(tmask.toBbox, rle_pt)
    # compare the results
    assert list(result_pt.numpy()) == result


@pytest.mark.benchmark(group="toBbox", warmup=True)
@parametrize_with_cases("mask, result", cases=BboxCases)
def test_toBbox_np(benchmark, mask: Tensor, result: list[int]):  # noqa: N802
    # create a mask
    mask_np = np.asfortranarray(mask.numpy())
    # compute the bounding box
    rle_np = nmask.encode(mask_np)
    result_np = benchmark(nmask.toBbox, rle_np)
    # compare the results
    assert list(result_np) == result


@parametrize_with_cases("mask, result", cases=BboxCases)
def test_toBbox(mask: Tensor, result: list[int]):  # noqa: N802
    # create a mask
    mask_pt = mask
    mask_np = np.asfortranarray(mask_pt.numpy())
    # compute the bounding box
    rle_np = nmask.encode(mask_np)
    rle_pt = tmask.encode(mask_pt)
    result_np = nmask.toBbox(rle_np)
    result_pt = tmask.toBbox(rle_pt)
    # compare the results
    assert np.all(result_np == result_pt.numpy())  # np.allclose(bbox1, bbox2.numpy())
    assert list(result_np) == result
