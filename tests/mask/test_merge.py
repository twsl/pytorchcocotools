import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_cases import case, parametrize_with_cases
from pytorchcocotools._maskApi import (
    RleObj,
    RleObjs,
)
import pytorchcocotools.mask as tmask
import torch
from torch import Tensor


class MergeCases:
    h = 25
    w = 25
    mask = torch.zeros((h, w), dtype=torch.uint8)

    def _build_mask(self, min1: int, max1: int) -> Tensor:
        mask_pt1 = self.mask.clone()
        mask_pt1[min1:max1, min1:max1] = 1
        return mask_pt1

    def case_start_area_same(self) -> tuple:
        return (
            self._build_mask(0, 5),
            self._build_mask(0, 5),
            False,
            {"size": [self.h, self.w], "counts": b"05d00000000d?"},
        )

    def case_center_area_same(self) -> tuple:
        return (
            self._build_mask(5, 10),
            self._build_mask(5, 10),
            False,
            {"size": [self.h, self.w], "counts": b"R45d00000000b;"},
        )

    def case_end_area_same(self) -> tuple:
        return (
            self._build_mask(20, 25),
            self._build_mask(20, 25),
            False,
            {"size": [self.h, self.w], "counts": b"X`05d00000000"},
        )

    def case_full_area_same(self) -> tuple:
        return (
            self._build_mask(0, 25),
            self._build_mask(0, 25),
            False,
            {"size": [self.h, self.w], "counts": b"0ac0"},
        )

    def case_center_area_none(self) -> tuple:
        return (
            self._build_mask(5, 10),
            self._build_mask(10, 15),
            False,
            {"size": [self.h, self.w], "counts": b"R45d0000000050K0000000`7"},
        )

    def case_center_area_partial(self) -> tuple:
        return (
            self._build_mask(5, 10),
            self._build_mask(5, 15),
            False,
            {"size": [self.h, self.w], "counts": b"R4:?00000000000000000e7"},
        )

    def case_center_area_overlap(self) -> tuple:
        return (
            self._build_mask(5, 10),
            self._build_mask(8, 13),
            False,
            {"size": [self.h, self.w], "counts": b"R45d000003M03M0000T9"},
        )


@pytest.mark.benchmark(group="merge", warmup=True)
@parametrize_with_cases("obj1, obj2, intersect, result", cases=MergeCases)
def test_merge_pt(benchmark, obj1: Tensor, obj2: Tensor, intersect: bool, result: RleObj):
    # encode
    rle_pt1 = tmask.encode(obj1)
    rle_pt2 = tmask.encode(obj2)
    # compute the iou
    result_pt = benchmark(tmask.merge, [rle_pt1, rle_pt2], intersect=intersect)
    # compare the results
    assert result_pt["counts"] == result["counts"]
    assert result_pt["size"] == result["size"]


@pytest.mark.benchmark(group="merge", warmup=True)
@parametrize_with_cases("obj1, obj2, intersect, result", cases=MergeCases)
def test_merge_np(benchmark, obj1: Tensor, obj2: Tensor, intersect: bool, result: RleObj):
    obj1 = np.asfortranarray(obj1.numpy())
    obj2 = np.asfortranarray(obj2.numpy())
    # encode
    rle_np1 = mask.encode(obj1)
    rle_np2 = mask.encode(obj2)
    # compute the iou
    result_np = benchmark(mask.merge, [rle_np1, rle_np2], intersect=intersect)
    # compare the results
    assert result_np["counts"] == result["counts"]
    assert result_np["size"] == result["size"]


@parametrize_with_cases("obj1, obj2, intersect, result", cases=MergeCases)
def test_merge(obj1: Tensor, obj2: Tensor, intersect: bool, result: RleObj):
    # create two masks
    mask_pt1 = obj1
    mask_pt2 = obj2
    mask_np1 = np.asfortranarray(mask_pt1.numpy())
    mask_np2 = np.asfortranarray(mask_pt2.numpy())
    # compute the iou
    rle_np1 = mask.encode(mask_np1)
    rle_np2 = mask.encode(mask_np2)
    rle_pt1 = tmask.encode(mask_pt1)
    rle_pt2 = tmask.encode(mask_pt2)
    # merge the masks
    merged_np = mask.merge([rle_np1, rle_np2], intersect=intersect)
    merged_pt = tmask.merge([rle_pt1, rle_pt2], intersect=intersect)
    # compare the results
    assert merged_np == merged_pt
    assert merged_pt["counts"] == merged_np["counts"]
    assert merged_pt["size"] == merged_np["size"]
    assert merged_pt["counts"] == result["counts"]
    assert merged_pt["size"] == result["size"]
