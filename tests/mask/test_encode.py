from typing import Any

import numpy as np
import pycocotools.mask as nmask
import pytest
from pytest_cases import parametrize_with_cases
import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RleObj
import pytorchcocotools.mask as tmask

from .base_cases import BaseCases


class EncodeCases(BaseCases):
    def case_start_area(self) -> tuple[Tensor, RleObj]:
        return (self._build_mask(0, 5), RleObj({"size": [self.h, self.w], "counts": b"05d00000000d?"}))

    def case_center_area(self) -> tuple[Tensor, RleObj]:
        return (self._build_mask(5, 10), RleObj({"size": [self.h, self.w], "counts": b"R45d00000000b;"}))

    def case_end_area(self) -> tuple[Tensor, RleObj]:
        return (self._build_mask(20, 25), RleObj({"size": [self.h, self.w], "counts": b"X`05d00000000"}))

    def case_full_area(self) -> tuple[Tensor, RleObj]:
        return (self._build_mask(0, 25), RleObj({"size": [self.h, self.w], "counts": b"0ac0"}))

    def case_complex_1_np(self) -> tuple[Tensor, RleObj]:
        h = 427
        w = 640
        data = RleObj(
            {
                "size": [h, w],
                "counts": b"\\`_3;j<6M3E_OjCd0T<:O1O2O001O00001O00001O001O0000O1K6J5J6A^C0g<N=O001O0O2Omk^4",
            }
        )
        return (
            torch.from_numpy(nmask.decode(data)),  # pyright: ignore[reportArgumentType]
            data,
        )

    def case_complex_1_pt(self) -> tuple[Tensor, RleObj]:
        h = 427
        w = 640
        data = RleObj(
            {
                "size": [h, w],
                "counts": b"\\`_3;j<6M3E_OjCd0T<:O1O2O001O00001O00001O001O0000O1K6J5J6A^C0g<N=O001O0O2Omk^4",
            }
        )
        return (
            tmask.decode(data),  # pyright: ignore[reportArgumentType]
            data,
        )

    def case_complex_2_np(self) -> tuple[Tensor, RleObj]:
        h = 427
        w = 640
        data = RleObj({"size": [h, w], "counts": b"RT_32n<<O100O0010O000010O0001O00001O000O101O0ISPc4"})

        return (
            torch.from_numpy(nmask.decode(data)),  # pyright: ignore[reportArgumentType]
            data,
        )

    def case_complex_2_pt(self) -> tuple[Tensor, RleObj]:
        h = 427
        w = 640
        data = RleObj({"size": [h, w], "counts": b"RT_32n<<O100O0010O000010O0001O00001O000O101O0ISPc4"})

        return (
            tmask.decode(data),  # pyright: ignore[reportArgumentType]
            data,
        )


@pytest.mark.benchmark(group="encode", warmup=True)
@parametrize_with_cases("mask, result", cases=EncodeCases)
def test_encode_pt(benchmark, mask: Tensor, result: RleObj) -> None:  # noqa: N802
    # create a mask
    mask_pt = mask
    # encode the mask
    result_pt: Tensor = benchmark(tmask.encode, mask_pt)
    # compare the results
    assert result_pt == result


@pytest.mark.benchmark(group="encode", warmup=True)
@parametrize_with_cases("mask, result", cases=EncodeCases)
def test_encode_np(benchmark, mask: Tensor, result: RleObj) -> None:  # noqa: N802
    # create a mask
    mask_np = np.asfortranarray(mask.numpy())
    # encode the mask
    result_np = benchmark(nmask.encode, mask_np)
    # compare the results
    assert result_np == result


@parametrize_with_cases("mask, result", cases=EncodeCases)
def test_encode(mask: Tensor, result: RleObj) -> None:  # noqa: N802
    # create a mask
    mask_pt = mask
    mask_np = np.asfortranarray(mask_pt.numpy())
    # encode the mask
    result_np = nmask.encode(mask_np)
    result_pt = tmask.encode(mask_pt)
    # compare the results
    assert result_np["counts"] == result_pt["counts"]
    assert result_np["size"] == result_pt["size"]
    assert result["counts"] == result_pt["counts"]
    assert result["size"] == result_pt["size"]
