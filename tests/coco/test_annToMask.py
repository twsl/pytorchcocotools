import numpy as np
from pycocotools.coco import COCO as COCO
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class AnnToMaskCases:
    def case_test(self) -> tuple:
        return (2096753, None)


@pytest.mark.benchmark(group="annToMask", warmup=True)
@parametrize_with_cases("img_id, result", cases=AnnToMaskCases)
def test_annToMask_pt(benchmark, coco_pt: COCOpt, img_id: int, result: int) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_pt = coco_pt.loadAnns(img_id)
    # get the mask for the annotation
    mask_pt = benchmark(coco_pt.annToMask(ann_pt[0]))
    # compare the results
    # np.nonzero(mask_np)
    assert np.array_equal(result, mask_pt.numpy())


@pytest.mark.benchmark(group="annToMask", warmup=True)
@parametrize_with_cases("img_id, result", cases=AnnToMaskCases)
def test_annToMask_np(benchmark, coco_np: COCO, img_id: int, result: int) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_np = coco_np.loadAnns(img_id)
    # get the mask for the annotation
    mask_np = benchmark(coco_np.annToMask(ann_np[0]))
    # compare the results
    assert np.array_equal(mask_np, result)


@parametrize_with_cases("img_id, result", cases=AnnToMaskCases)
def test_annToMask(coco_np: COCO, coco_pt: COCOpt, img_id: int, result) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_np = coco_np.loadAnns(img_id)
    ann_pt = coco_pt.loadAnns(img_id)
    # get the mask for the annotation
    mask_np = coco_np.annToMask(ann_np[0])
    mask_pt = coco_pt.annToMask(ann_pt[0])
    # compare the results
    # np.nonzero(mask_np)
    assert np.array_equal(mask_np, mask_pt.numpy())
    assert np.array_equal(mask_np, result)
