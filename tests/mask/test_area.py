import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torchvision import tv_tensors as tv

import pytorchcocotools.mask as tmask


class AreaCases:
    h = 25
    w = 25

    # @case(id="start_area")
    def case_start_area(self) -> tuple[int, int, int, int, int]:
        return (0, 5, self.h, self.w, 25)

    def case_center_area(self) -> tuple[int, int, int, int, int]:
        return (5, 10, self.h, self.w, 25)

    def case_end_area(self) -> tuple[int, int, int, int, int]:
        return (20, 25, self.h, self.w, 25)

    def case_full_area(self) -> tuple[int, int, int, int, int]:
        return (0, 25, self.h, self.w, 625)


@pytest.mark.benchmark(group="area", warmup=True)
@parametrize_with_cases("min, max, h, w, result", cases=AreaCases)
def test_area_pt(benchmark: BenchmarkFixture, device: str, min: int, max: int, h: int, w: int, result: int) -> None:
    # create a mask
    mask_pt = tv.Mask(torch.zeros((h, w), dtype=torch.uint8, device=device))
    mask_pt[min:max, min:max] = 1
    # compute the area
    rle_pt = tmask.encode(mask_pt, device=device)
    with torch.no_grad():
        result_pt = benchmark(tmask.area, rle_pt[0], device=device)
    # compare the results
    assert result_pt[0] == result


@pytest.mark.benchmark(group="area", warmup=True)
@parametrize_with_cases("min, max, h, w, result", cases=AreaCases)
def test_area_np(benchmark: BenchmarkFixture, device: str, min: int, max: int, h: int, w: int, result: int) -> None:
    # create a mask
    mask_np = np.zeros((h, w), dtype=np.uint8, order="F")
    mask_np[min:max, min:max] = 1
    # compute the area
    rle_np = mask.encode(mask_np)
    result_np = benchmark(mask.area, rle_np)
    # compare the results
    assert result_np == result


@parametrize_with_cases("min, max, h, w, result", cases=AreaCases)
def test_area(device: str, min: int, max: int, h: int, w: int, result: int) -> None:
    # create a mask
    mask_pt = tv.Mask(torch.zeros((h, w), dtype=torch.uint8, device=device))
    mask_pt[min:max, min:max] = 1
    mask_np = np.asfortranarray(mask_pt.cpu().numpy())
    # compute the area
    rle_np = mask.encode(mask_np)
    rle_pt = tmask.encode(mask_pt, device=device)
    result_np = mask.area(rle_np)
    result_pt = tmask.area(rle_pt, device=device)
    # compare the results
    assert result_np == result_pt
    assert result_np == result
