from typing import cast

import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RleObj, RleObjs
import pytorchcocotools.mask as tmask


class IoUCases:
    h = 25
    w = 25
    mask = torch.zeros((h, w), dtype=torch.uint8)

    def _build_mask(self, min1: int, max1: int) -> tv.Mask:
        mask_pt1 = self.mask.clone()
        mask_pt1[min1:max1, min1:max1] = 1  # xyxy
        return tv.Mask(mask_pt1)

    def _build_bbox(self, min1: int, max1: int) -> tv.BoundingBoxes:
        return tv.BoundingBoxes(
            torch.tensor([min1, min1, max1 - min1, max1 - min1], dtype=torch.int32).unsqueeze(0),
            format=tv.BoundingBoxFormat.XYWH,
            canvas_size=(self.h, self.w),
        )  # pyright: ignore[reportCallIssue]

    def case_start_area_same(self) -> tuple[tv.Mask, tv.Mask, list[bool], float]:
        return (self._build_mask(0, 5), self._build_mask(0, 5), [False], 1)

    def case_center_area_same(self) -> tuple[tv.Mask, tv.Mask, list[bool], float]:
        return (self._build_mask(5, 10), self._build_mask(5, 10), [False], 1)

    def case_end_area_same(self) -> tuple[tv.Mask, tv.Mask, list[bool], float]:
        return (self._build_mask(20, 25), self._build_mask(20, 25), [False], 1)

    def case_full_area_same(self) -> tuple[tv.Mask, tv.Mask, list[bool], float]:
        return (self._build_mask(0, 25), self._build_mask(0, 25), [False], 1)

    def case_center_area_none(self) -> tuple[tv.Mask, tv.Mask, list[bool], float]:
        return (self._build_mask(5, 10), self._build_mask(10, 15), [False], 0)

    def case_center_area_partial(self) -> tuple[tv.Mask, tv.Mask, list[bool], float]:
        return (self._build_mask(5, 10), self._build_mask(5, 15), [False], 0.25)

    def case_center_area_overlap(self) -> tuple[tv.Mask, tv.Mask, list[bool], float]:
        return (self._build_mask(5, 10), self._build_mask(8, 13), [False], 4 / 46)  # 0.08695652

    def case_bbox_start_area_same(self) -> tuple[tv.BoundingBoxes, tv.BoundingBoxes, list[bool], float]:
        return (self._build_bbox(0, 5), self._build_bbox(0, 5), [False], 1)

    def case_bbox_center_area_same(self) -> tuple[tv.BoundingBoxes, tv.BoundingBoxes, list[bool], float]:
        return (self._build_bbox(5, 10), self._build_bbox(5, 10), [False], 1)

    def case_bbox_end_area_same(self) -> tuple[tv.BoundingBoxes, tv.BoundingBoxes, list[bool], float]:
        return (self._build_bbox(20, 25), self._build_bbox(20, 25), [False], 1)

    def case_bbox_full_area_same(self) -> tuple[tv.BoundingBoxes, tv.BoundingBoxes, list[bool], float]:
        return (self._build_bbox(0, 25), self._build_bbox(0, 25), [False], 1)

    def case_bbox_center_area_none(self) -> tuple[tv.BoundingBoxes, tv.BoundingBoxes, list[bool], float]:
        return (self._build_bbox(5, 10), self._build_bbox(10, 15), [False], 0)

    def case_bbox_center_area_partial(self) -> tuple[tv.BoundingBoxes, tv.BoundingBoxes, list[bool], float]:
        return (self._build_bbox(5, 10), self._build_bbox(5, 15), [False], 0.25)

    def case_bbox_center_area_overlap(self) -> tuple[tv.BoundingBoxes, tv.BoundingBoxes, list[bool], float]:
        return (self._build_bbox(5, 10), self._build_bbox(8, 13), [False], 4 / 46)  # 0.08695652


@pytest.mark.benchmark(group="iou", warmup=True)
@parametrize_with_cases("obj1, obj2, iscrowd, result", cases=IoUCases)
def test_iou_pt(
    benchmark: BenchmarkFixture,
    obj1: tv.Mask | tv.BoundingBoxes,
    obj2: tv.Mask | tv.BoundingBoxes,
    iscrowd: list[bool],
    result: float,
) -> None:
    # encode
    if isinstance(obj1, tv.Mask) and isinstance(obj2, tv.Mask):
        obj1_ = tmask.encode(obj1)
        obj2_ = tmask.encode(obj2)
    else:
        obj1_ = obj1
        obj2_ = obj2
    # compute the iou
    result_pt = benchmark(tmask.iou, obj1_, obj2_, iscrowd)
    # compare the results
    assert result_pt == result


@pytest.mark.benchmark(group="iou", warmup=True)
@parametrize_with_cases("obj1, obj2, iscrowd, result", cases=IoUCases)
def test_iou_np(
    benchmark: BenchmarkFixture,
    obj1: tv.Mask | tv.BoundingBoxes,
    obj2: tv.Mask | tv.BoundingBoxes,
    iscrowd: list[bool],
    result: float,
) -> None:
    obj1n = np.asfortranarray(obj1.numpy())
    obj2n = np.asfortranarray(obj2.numpy())
    # encode
    if isinstance(obj1, tv.Mask) and isinstance(obj2, tv.Mask):
        rle_np1 = mask.encode(obj1n)
        rle_np2 = mask.encode(obj2n)
        obj1_ = [rle_np1]
        obj2_ = [rle_np2]
    else:
        obj1_ = obj1n
        obj2_ = obj2n
    # compute the iou
    result_np = benchmark(mask.iou, obj1_, obj2_, [int(c) for c in iscrowd])
    # compare the results
    assert result_np == result


@parametrize_with_cases("obj1, obj2, iscrowd, result", cases=IoUCases)
def test_iou(
    obj1: tv.Mask | tv.BoundingBoxes, obj2: tv.Mask | tv.BoundingBoxes, iscrowd: list[bool], result: float
) -> None:
    # create two masks
    mask_pt1 = obj1
    mask_pt2 = obj2
    mask_np1 = np.asfortranarray(obj1.numpy())
    mask_np2 = np.asfortranarray(obj2.numpy())
    # encode the masks
    if isinstance(mask_pt1, tv.Mask) and isinstance(mask_pt2, tv.Mask):
        rle_np1 = mask.encode(mask_np1)
        rle_np2 = mask.encode(mask_np2)
        rle_pt1 = tmask.encode(mask_pt1)[0]
        rle_pt2 = tmask.encode(mask_pt2)[0]
        # make RleObjs/lists
        obj_np1 = [rle_np1]
        obj_np2 = [rle_np2]
        obj_pt1 = [rle_pt1]
        obj_pt2 = [rle_pt2]
    else:
        obj_np1 = mask_np1
        obj_np2 = mask_np2
        obj_pt1 = mask_pt1
        obj_pt2 = mask_pt2

    # compute the iou
    iscrowd = [bool(c) for c in iscrowd]
    iou_np = mask.iou(obj_np1, obj_np2, iscrowd)
    iou_pt = tmask.iou(obj_pt1, obj_pt2, iscrowd)  # pyright: ignore[reportArgumentType]
    # compare the results
    assert iou_np[0][0] == iou_pt[0][0]
    assert iou_np[0][0] == result
