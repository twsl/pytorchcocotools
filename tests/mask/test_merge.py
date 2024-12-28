from typing import cast

import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_cases import parametrize_with_cases
import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RleObj
import pytorchcocotools.mask as tmask


class MergeCases:
    h = 25
    w = 25
    mask = torch.zeros((h, w), dtype=torch.uint8)

    def _build_mask(self, min1: int, max1: int) -> Tensor:
        mask_pt1 = self.mask.clone()
        mask_pt1[min1:max1, min1:max1] = 1
        return mask_pt1

    def case_start_area_same(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(0, 5),
            self._build_mask(0, 5),
            False,
            RleObj(size=[self.h, self.w], counts=b"05d00000000d?"),
        )

    def case_center_area_same(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(5, 10),
            self._build_mask(5, 10),
            False,
            RleObj(size=[self.h, self.w], counts=b"R45d00000000b;"),
        )

    def case_end_area_same(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(20, 25),
            self._build_mask(20, 25),
            False,
            RleObj(size=[self.h, self.w], counts=b"X`05d00000000"),
        )

    def case_full_area_same(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(0, 25),
            self._build_mask(0, 25),
            False,
            RleObj(size=[self.h, self.w], counts=b"0ac0"),
        )

    def case_center_area_none(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(5, 10),
            self._build_mask(10, 15),
            False,
            RleObj(size=[self.h, self.w], counts=b"R45d0000000050K0000000`7"),
        )

    def case_center_area_partial(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(5, 10),
            self._build_mask(5, 15),
            False,
            RleObj(size=[self.h, self.w], counts=b"R4:?00000000000000000e7"),
        )

    def case_center_area_overlap(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(5, 10),
            self._build_mask(8, 13),
            False,
            RleObj(size=[self.h, self.w], counts=b"R45d000003M03M0000T9"),
        )

    def case_start_area_same_intersect(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(0, 5),
            self._build_mask(0, 5),
            True,
            RleObj(size=[self.h, self.w], counts=b"05d00000000d?"),
        )

    def case_center_area_same_intersect(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(5, 10),
            self._build_mask(5, 10),
            True,
            RleObj(size=[self.h, self.w], counts=b"R45d00000000b;"),
        )

    def case_end_area_same_intersect(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(20, 25),
            self._build_mask(20, 25),
            True,
            RleObj(size=[self.h, self.w], counts=b"X`05d00000000"),
        )

    def case_full_area_same_intersect(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(0, 25),
            self._build_mask(0, 25),
            True,
            RleObj(size=[self.h, self.w], counts=b"0ac0"),
        )

    def case_center_area_none_intersect(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(5, 10),
            self._build_mask(10, 15),
            True,
            RleObj(size=[self.h, self.w], counts=b"ac0"),
        )

    def case_center_area_partial_intersect(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(5, 10),
            self._build_mask(5, 15),
            True,
            RleObj(size=[self.h, self.w], counts=b"R45d00000000b;"),
        )

    def case_center_area_overlap_intersect(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        return (
            self._build_mask(5, 10),
            self._build_mask(8, 13),
            True,
            RleObj(size=[self.h, self.w], counts=b"`62g00_;"),
        )

    def case_complex(self) -> tuple[Tensor, Tensor, bool, RleObj]:
        from pytorchcocotools.mask import decode

        h = 427
        w = 640
        data1 = RleObj(
            size=[h, w],
            counts=b"\\`_3;j<6M3E_OjCd0T<:O1O2O001O00001O00001O001O0000O1K6J5J6A^C0g<N=O001O0O2Omk^4",
        )
        data2 = RleObj(size=[h, w], counts=b"RT_32n<<O100O0010O000010O0001O00001O000O101O0ISPc4")

        return (
            decode(data1).squeeze(-1),
            decode(data2).squeeze(-1),
            False,
            RleObj(
                size=[h, w],
                counts=b"RT_32X<9SD3f;3ZDNb;5_DKU;DeDd05HU;b0kD_OS;b0nD]OQ;e0nD[OR;e0nD[OR;f0nDZOQ;f0oDZOQ;f0oDZOQ;g0oDXOQ;h0oDXOQ;h0oDXOQ;i0oDVOQ;j0oDVOQ;k0nDTOS;l0mDTOS;l0nDSOR;l0oDmNX;n0f0J5J6A^C0g<N=O001O0O2Omk^4",
            ),
        )


@pytest.mark.benchmark(group="merge", warmup=True)
@parametrize_with_cases("obj1, obj2, intersect, result", cases=MergeCases)
def test_merge_pt(benchmark, obj1: Tensor, obj2: Tensor, intersect: bool, result: RleObj) -> None:
    # encode
    rle_pt1 = tmask.encode(tv.Mask(obj1))
    rle_pt2 = tmask.encode(tv.Mask(obj2))
    # compute the iou
    result_pt = cast(RleObj, benchmark(tmask.merge, [rle_pt1[0], rle_pt2[0]], intersect=intersect))
    # compare the results
    assert result_pt.counts == result.counts
    assert result_pt.size == result.size


@pytest.mark.benchmark(group="merge", warmup=True)
@parametrize_with_cases("obj1, obj2, intersect, result", cases=MergeCases)
def test_merge_np(benchmark, obj1: Tensor, obj2: Tensor, intersect: bool, result: RleObj) -> None:
    obj1n = np.asfortranarray(obj1.numpy())
    obj2n = np.asfortranarray(obj2.numpy())
    # encode
    rle_np1 = mask.encode(obj1n)
    rle_np2 = mask.encode(obj2n)
    # compute the iou
    result_np = benchmark(mask.merge, [rle_np1, rle_np2], intersect=intersect)
    # compare the results
    assert result_np["counts"] == result.counts
    assert result_np["size"] == result.size


@parametrize_with_cases("obj1, obj2, intersect, result", cases=MergeCases)
def test_merge(obj1: Tensor, obj2: Tensor, intersect: bool, result: RleObj) -> None:
    # create two masks
    mask_pt1 = tv.Mask(obj1)
    mask_pt2 = tv.Mask(obj2)
    mask_np1 = np.asfortranarray(mask_pt1.numpy())
    mask_np2 = np.asfortranarray(mask_pt2.numpy())
    # compute the iou
    rle_np1 = mask.encode(mask_np1)
    rle_np2 = mask.encode(mask_np2)
    rle_pt1 = tmask.encode(mask_pt1)[0]
    rle_pt2 = tmask.encode(mask_pt2)[0]
    # merge the masks
    merged_np = mask.merge([rle_np1, rle_np2], intersect=intersect)
    merged_pt = tmask.merge([rle_pt1, rle_pt2], intersect=intersect)
    # compare the results
    assert merged_np == merged_pt.__dict__
    assert merged_pt.counts == merged_np["counts"]
    assert merged_pt.size == merged_np["size"]
    assert merged_pt.counts == result.counts
    assert merged_pt.size == result.size
