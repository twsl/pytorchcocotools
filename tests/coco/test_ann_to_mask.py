from typing import cast

import numpy as np
from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class AnnToMaskCases:
    def case_test_1(self) -> tuple[int, np.ndarray]:
        from pycocotools.mask import decode

        data = {
            "size": [427, 640],
            "counts": b"RT_32X<9SD3f;3ZDNb;5_DKU;DeDd05HU;b0kD_OS;b0nD]OQ;e0nD[OR;e0nD[OR;f0nDZOQ;f0oDZOQ;f0oDZOQ;g0oDXOQ;h0oDXOQ;h0oDXOQ;i0oDVOQ;j0oDVOQ;k0nDTOS;l0mDTOS;l0nDSOR;l0oDmNX;n0f0J5J6A^C0g<N=O001O0O2Omk^4",
        }
        mask = decode(data)  # pyright: ignore[reportArgumentType]
        return (2096753, mask)


@pytest.mark.benchmark(group="annToMask", warmup=True)
@parametrize_with_cases("img_ids, result", cases=AnnToMaskCases)
def test_annToMask_pt(benchmark: BenchmarkFixture, coco_pt: COCOpt, img_ids: int, result) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_pt = coco_pt.loadAnns(img_ids)
    # get the mask for the annotation
    mask_pt = cast(tv.Mask, benchmark(coco_pt.annToMask, ann_pt[0]))
    # compare the results
    # np.nonzero(mask_np)
    assert np.array_equal(result, mask_pt.squeeze(-1).numpy())


@pytest.mark.benchmark(group="annToMask", warmup=True)
@parametrize_with_cases("img_ids, result", cases=AnnToMaskCases)
def test_annToMask_np(benchmark: BenchmarkFixture, coco_np: COCOnp, img_ids: int, result) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_np = coco_np.loadAnns(img_ids)
    # get the mask for the annotation
    mask_np = cast(np.ndarray, benchmark(coco_np.annToMask, ann_np[0]))
    # compare the results
    assert np.array_equal(mask_np, result)


@parametrize_with_cases("img_ids, result", cases=AnnToMaskCases)
def test_annToMask(coco_np: COCOnp, coco_pt: COCOpt, img_ids: int, result) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_np = coco_np.loadAnns(img_ids)
    ann_pt = coco_pt.loadAnns(img_ids)
    # get the mask for the annotation
    mask_np = coco_np.annToMask(ann_np[0])
    mask_pt = coco_pt.annToMask(ann_pt[0])
    # compare the results
    # np.nonzero(mask_np)
    assert np.array_equal(mask_np, mask_pt.squeeze(-1).numpy())
    assert np.array_equal(mask_np, result)
