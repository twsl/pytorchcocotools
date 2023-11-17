import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_cases import case, parametrize_with_cases
import pytorchcocotools.mask as tmask
import torch


class AreaCases:
    h = 25
    w = 25

    # @case(id="start_area")
    def case_start_area(self):
        return (0, 5, self.h, self.w, 25)

    def case_center_area(self):
        return (5, 10, self.h, self.w, 25)

    def case_end_area(self):
        return (20, 25, self.h, self.w, 25)

    def case_full_area(self):
        return (0, 25, self.h, self.w, 625)


@pytest.mark.benchmark(group="area", warmup=True)
@parametrize_with_cases("min, max, h, w, area", cases=AreaCases)
def test_area_pt(benchmark, min: int, max: int, h: int, w: int, area: int):
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    # compute the area
    rle_pt = tmask.encode(mask_pt)
    result_pt = benchmark(tmask.area, rle_pt)
    # compare the results
    assert result_pt == area


@pytest.mark.benchmark(group="area", warmup=True)
@parametrize_with_cases("min, max, h, w, area", cases=AreaCases)
def test_area_np(benchmark, min: int, max: int, h: int, w: int, area: int):
    # create a mask
    mask_np = np.zeros((h, w), dtype=np.uint8, order="F")
    mask_np[min:max, min:max] = 1
    # compute the area
    rle_np = mask.encode(mask_np)
    result_np = benchmark(mask.area, rle_np)
    # compare the results
    assert result_np == area


@parametrize_with_cases("min, max, h, w, area", cases=AreaCases)
def test_area(min: int, max: int, h: int, w: int, area: int):
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    mask_np = np.asfortranarray(mask_pt.numpy())
    # compute the area
    rle_np = mask.encode(mask_np)
    rle_pt = tmask.encode(mask_pt)
    result_np = mask.area(rle_np)
    result_pt = tmask.area(rle_pt)
    # compare the results
    assert result_np == result_pt
    assert result_np == area
