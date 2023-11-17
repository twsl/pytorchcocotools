import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_cases import case, parametrize_with_cases
import pytorchcocotools.mask as tmask
import torch
from torch import Tensor


class BboxCases:
    h = 25
    w = 25

    # @case(id="start_area")
    def case_start_area(self):
        return (0, 5, self.h, self.w, [0, 0, 5, 5])

    def case_center_area(self):
        return (5, 10, self.h, self.w, [5, 5, 5, 5])

    def case_end_area(self):
        return (20, 25, self.h, self.w, [20, 20, 5, 5])

    def case_full_area(self):
        return (0, 25, self.h, self.w, [0, 0, 25, 25])


@pytest.mark.benchmark(group="toBbox", warmup=True)
@parametrize_with_cases("min, max, h, w, box", cases=BboxCases)
def test_toBbox_pt(benchmark, min: int, max: int, h: int, w: int, box: list[int]):  # noqa: N802
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    # compute the bounding box
    rle_pt = tmask.encode(mask_pt)
    result_pt: Tensor = benchmark(tmask.toBbox, rle_pt)
    # compare the results
    assert list(result_pt.numpy()) == box


@pytest.mark.benchmark(group="toBbox", warmup=True)
@parametrize_with_cases("min, max, h, w, box", cases=BboxCases)
def test_toBbox_np(benchmark, min: int, max: int, h: int, w: int, box: list[int]):  # noqa: N802
    # create a mask
    mask_np = np.zeros((h, w), dtype=np.uint8, order="F")
    mask_np[min:max, min:max] = 1
    # compute the bounding box
    rle_np = mask.encode(mask_np)
    result_np = benchmark(mask.toBbox, rle_np)
    # compare the results
    assert list(result_np) == box


@parametrize_with_cases("min, max, h, w, box", cases=BboxCases)
def test_toBbox(min: int, max: int, h: int, w: int, box: list[int]):  # noqa: N802
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    mask_np = np.asfortranarray(mask_pt.numpy())
    # compute the bounding box
    rle_np = mask.encode(mask_np)
    rle_pt = tmask.encode(mask_pt)
    result_np = mask.toBbox(rle_np)
    result_pt = tmask.toBbox(rle_pt)
    # compare the results
    assert np.all(result_np == result_pt.numpy())  # np.allclose(bbox1, bbox2.numpy())
    assert list(result_np) == box
