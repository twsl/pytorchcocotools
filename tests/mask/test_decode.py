import numpy as np
import pycocotools.mask as nmask
import pytest
from pytest_cases import parametrize_with_cases
import torch
from torch import Tensor

import pytorchcocotools.mask as tmask

from .base_cases import BaseCases


class DecodeCases(BaseCases):
    def case_start_area(self):
        mask_pt = self._build_mask(0, 5)
        return (mask_pt, mask_pt.clone())

    def case_center_area(self):
        mask_pt = self._build_mask(5, 10)
        return (mask_pt, mask_pt.clone())

    def case_end_area(self):
        mask_pt = self._build_mask(20, 25)
        return (mask_pt, mask_pt.clone())

    def case_full_area(self):
        mask_pt = self._build_mask(0, 25)
        return (mask_pt, mask_pt.clone())

    def case_complex_1_np(self) -> tuple:
        h = 427
        w = 640
        data = {
            "size": [h, w],
            "counts": b"\\`_3;j<6M3E_OjCd0T<:O1O2O001O00001O00001O001O0000O1K6J5J6A^C0g<N=O001O0O2Omk^4",
        }
        mask_pt = torch.from_numpy(nmask.decode(data))
        return (mask_pt, mask_pt.clone())

    def case_complex_1_pt(self) -> tuple:
        h = 427
        w = 640
        data = {
            "size": [h, w],
            "counts": b"\\`_3;j<6M3E_OjCd0T<:O1O2O001O00001O00001O001O0000O1K6J5J6A^C0g<N=O001O0O2Omk^4",
        }
        mask_pt = tmask.decode(data)
        return (mask_pt, mask_pt.clone())

    def case_complex_2_np(self) -> tuple:
        h = 427
        w = 640
        data = {"size": [h, w], "counts": b"RT_32n<<O100O0010O000010O0001O00001O000O101O0ISPc4"}

        mask_pt = torch.from_numpy(nmask.decode(data))
        return (mask_pt, mask_pt.clone())

    def case_complex_2_pt(self) -> tuple:
        h = 427
        w = 640
        data = {"size": [h, w], "counts": b"RT_32n<<O100O0010O000010O0001O00001O000O101O0ISPc4"}

        mask_pt = tmask.decode(data)
        return (mask_pt, mask_pt.clone())


@pytest.mark.benchmark(group="decode", warmup=True)
@parametrize_with_cases("mask, result", cases=DecodeCases)
def test_decode_pt(benchmark, mask: Tensor, result: Tensor):  # noqa: N802
    # create a mask
    mask_pt = mask
    # decode the mask
    rle_pt = tmask.encode(mask_pt)
    result_pt: Tensor = benchmark(tmask.decode, rle_pt)
    # compare the results
    assert torch.equal(result_pt, result)


@pytest.mark.benchmark(group="decode", warmup=True)
@parametrize_with_cases("mask, result", cases=DecodeCases)
def test_decode_np(benchmark, mask: Tensor, result: Tensor):  # noqa: N802
    # create a mask
    mask_np = np.asfortranarray(mask.numpy())
    # decode the mask
    rle_np = nmask.encode(mask_np)
    result_np = benchmark(nmask.decode, rle_np)
    # compare the results
    assert np.array_equal(result_np, result.numpy())


@parametrize_with_cases("mask, result", cases=DecodeCases)
def test_decode(mask: Tensor, result: Tensor):  # noqa: N802
    # create a mask
    mask_pt = mask
    mask_np = np.asfortranarray(mask_pt.numpy())
    # decode the mask
    rle_np = nmask.encode(mask_np)
    rle_pt = tmask.encode(mask_pt)
    result_np = nmask.decode(rle_np)
    result_pt = tmask.decode(rle_pt)
    # compare the results
    assert np.array_equal(result_np, mask_np)
    assert np.array_equal(result_np, mask_pt.numpy())
    assert np.array_equal(result_np, result_pt.numpy())
