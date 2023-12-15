import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_cases import case, parametrize_with_cases
import pytorchcocotools.mask as tmask
import torch
from torch import Tensor


class EncodeCases:
    h = 25
    w = 25

    # @case(id="start_area")
    def case_start_area(self):
        return (0, 5, self.h, self.w, {"size": [self.h, self.w], "counts": b"05d00000000d?"})

    def case_center_area(self):
        return (5, 10, self.h, self.w, {"size": [self.h, self.w], "counts": b"R45d00000000b;"})

    def case_end_area(self):
        return (20, 25, self.h, self.w, {"size": [self.h, self.w], "counts": b"X`05d00000000"})

    def case_full_area(self):
        return (0, 25, self.h, self.w, {"size": [self.h, self.w], "counts": b"0ac0"})


@pytest.mark.benchmark(group="encode", warmup=True)
@parametrize_with_cases("min, max, h, w, result", cases=EncodeCases)
def test_encode_pt(benchmark, min: int, max: int, h: int, w: int, result: dict):  # noqa: N802
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    # encode the mask
    result_pt: Tensor = benchmark(tmask.encode, mask_pt)
    # compare the results
    assert result_pt == result


@pytest.mark.benchmark(group="encode", warmup=True)
@parametrize_with_cases("min, max, h, w, result", cases=EncodeCases)
def test_encode_np(benchmark, min: int, max: int, h: int, w: int, result: dict):  # noqa: N802
    # create a mask
    mask_np = np.zeros((h, w), dtype=np.uint8, order="F")
    mask_np[min:max, min:max] = 1
    # encode the mask
    result_np = benchmark(mask.encode, mask_np)
    # compare the results
    assert result_np == result


@parametrize_with_cases("min, max, h, w, result", cases=EncodeCases)
def test_encode(min: int, max: int, h: int, w: int, result: dict):  # noqa: N802
    # create a mask
    mask_pt = torch.zeros((h, w), dtype=torch.uint8)
    mask_pt[min:max, min:max] = 1
    mask_np = np.asfortranarray(mask_pt.numpy())
    # encode the mask
    result_np = mask.encode(mask_np)
    result_pt = tmask.encode(mask_pt)
    # compare the results
    assert result_np["counts"] == result_pt["counts"]
    assert result_np["size"] == result_pt["size"]
    assert result["counts"] == result_pt["counts"]
    assert result["size"] == result_pt["size"]
