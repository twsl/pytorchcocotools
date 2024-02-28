import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools._maskApi import (
    RleObjs,
)
import pytorchcocotools.mask as tmask
import torch
from torch import Tensor


class IoUCases:
    h = 25
    w = 25
    mask = torch.zeros((h, w), dtype=torch.uint8)

    def _build_mask(self, min1: int, max1: int) -> Tensor:
        mask_pt1 = self.mask.clone()
        mask_pt1[min1:max1, min1:max1] = 1
        return mask_pt1

    def _build_bbox(self, min1: int, max1: int) -> Tensor:
        return torch.tensor([min1, min1, max1 - min1, max1 - min1], dtype=torch.int32)

    def case_start_area_same(self) -> tuple:
        return (self._build_mask(0, 5), self._build_mask(0, 5), [False], True, 1)

    def case_center_area_same(self) -> tuple:
        return (self._build_mask(5, 10), self._build_mask(5, 10), [False], True, 1)

    def case_end_area_same(self) -> tuple:
        return (self._build_mask(20, 25), self._build_mask(20, 25), [False], True, 1)

    def case_full_area_same(self) -> tuple:
        return (self._build_mask(0, 25), self._build_mask(0, 25), [False], True, 1)

    def case_center_area_none(self) -> tuple:
        return (self._build_mask(5, 10), self._build_mask(10, 15), [False], True, 0)

    def case_center_area_partial(self) -> tuple:
        return (self._build_mask(5, 10), self._build_mask(5, 15), [False], True, 0.25)

    def case_center_area_overlap(self) -> tuple:
        return (self._build_mask(5, 10), self._build_mask(8, 13), [False], True, 4 / 46)  # 0.08695652

    def case_bbox_start_area_same(self) -> tuple:
        return (self._build_bbox(0, 5), self._build_bbox(0, 5), [False], False, 1)

    def case_bbox_center_area_same(self) -> tuple:
        return (self._build_bbox(5, 10), self._build_bbox(5, 10), [False], False, 1)

    def case_bbox_end_area_same(self) -> tuple:
        return (self._build_bbox(20, 25), self._build_bbox(20, 25), [False], False, 1)

    def case_bbox_full_area_same(self) -> tuple:
        return (self._build_bbox(0, 25), self._build_bbox(0, 25), [False], False, 1)

    def case_bbox_center_area_none(self) -> tuple:
        return (self._build_bbox(5, 10), self._build_bbox(10, 15), [False], False, 0)

    def case_bbox_center_area_partial(self) -> tuple:
        return (self._build_bbox(5, 10), self._build_bbox(5, 15), [False], False, 0.25)

    def case_bbox_center_area_overlap(self) -> tuple:
        return (self._build_bbox(5, 10), self._build_bbox(8, 13), [False], False, 4 / 46)  # 0.08695652


@pytest.mark.benchmark(group="iou", warmup=True)
@parametrize_with_cases("obj1, obj2, iscrowd, encode, result", cases=IoUCases)
def test_iou_pt(benchmark, obj1: Tensor, obj2: Tensor, iscrowd: list[bool], encode: bool, result: float):
    # encode
    if encode:
        rle_pt1 = tmask.encode(obj1)
        rle_pt2 = tmask.encode(obj2)
        obj1 = RleObjs([rle_pt1])
        obj2 = RleObjs([rle_pt2])
    else:
        obj1 = obj1.unsqueeze(0)
        obj2 = obj2.unsqueeze(0)
    # compute the iou
    result_pt = benchmark(tmask.iou, obj1, obj2, iscrowd)
    # compare the results
    assert result_pt == result


@pytest.mark.benchmark(group="iou", warmup=True)
@parametrize_with_cases("obj1, obj2, iscrowd, encode, result", cases=IoUCases)
def test_iou_np(benchmark, obj1: Tensor, obj2: Tensor, iscrowd: list[bool], encode: bool, result: float):
    obj1 = np.asfortranarray(obj1.numpy())
    obj2 = np.asfortranarray(obj2.numpy())
    # encode
    if encode:
        rle_np1 = mask.encode(obj1)
        rle_np2 = mask.encode(obj2)
        obj1 = [rle_np1]
        obj2 = [rle_np2]
    else:
        obj1 = obj1[np.newaxis, ...]
        obj2 = obj2[np.newaxis, ...]
    # compute the iou
    result_np = benchmark(mask.iou, obj1, obj2, [int(c) for c in iscrowd])
    # compare the results
    assert result_np == result


@parametrize_with_cases("obj1, obj2, iscrowd, encode, result", cases=IoUCases)
def test_iou(obj1: Tensor, obj2: Tensor, iscrowd: list[bool], encode: bool, result: float):
    # create two masks
    mask_pt1 = obj1
    mask_pt2 = obj2
    mask_np1 = np.asfortranarray(obj1.numpy())
    mask_np2 = np.asfortranarray(obj2.numpy())
    # encode the masks
    if encode:
        rle_np1 = mask.encode(mask_np1)
        rle_np2 = mask.encode(mask_np2)
        rle_pt1 = tmask.encode(mask_pt1)
        rle_pt2 = tmask.encode(mask_pt2)
        # make RleObjs/lists
        obj_np1 = [rle_np1]
        obj_np2 = [rle_np2]
        obj_pt1 = RleObjs([rle_pt1])
        obj_pt2 = RleObjs([rle_pt2])
    else:
        obj_np1 = mask_np1[np.newaxis, ...]
        obj_np2 = mask_np2[np.newaxis, ...]
        obj_pt1 = mask_pt1.unsqueeze(0)
        obj_pt2 = mask_pt2.unsqueeze(0)

    # compute the iou
    iscrowd = [int(c) for c in iscrowd]
    iou_np = mask.iou(obj_np1, obj_np2, iscrowd)
    iou_pt = tmask.iou(obj_pt1, obj_pt2, iscrowd)
    # compare the results
    assert iou_np[0][0] == iou_pt[0][0]
    assert iou_np[0][0] == result
