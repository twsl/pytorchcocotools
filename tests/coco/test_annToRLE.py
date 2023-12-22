import numpy as np
from pycocotools.coco import COCO as COCO
import pytest
from pytest_cases import case, fixture, parametrize, parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class AnnToRLECases:
    def case_test(self, coco1: COCO, coco2: COCOpt) -> tuple:
        return (coco1, coco2, 2096753, None)


@pytest.mark.benchmark(group="annToRLE", warmup=True)
@parametrize_with_cases("img_id, result", cases=AnnToRLECases)
def test_annToRLE_pt(benchmark, coco_pt: COCOpt, ann_id: int, result) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_pt = coco_pt.loadAnns(ann_id)
    # get the mask for the annotation
    rle_pt = benchmark(coco_pt.annToRLE(ann_pt[0]))
    # compare the results
    assert rle_pt["counts"] == result["counts"]
    assert rle_pt["size"] == result["size"]


@pytest.mark.benchmark(group="annToRLE", warmup=True)
@parametrize_with_cases("img_id, result", cases=AnnToRLECases)
def test_annToRLE_np(benchmark, coco_np: COCO, ann_id: int, result) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_np = coco_np.loadAnns(ann_id)
    # get the mask for the annotation
    rle_np = benchmark(coco_np.annToRLE(ann_np[0]))
    # compare the results
    assert rle_np["counts"] == result["counts"]
    assert rle_np["size"] == result["size"]


@parametrize_with_cases("img_id, result", cases=AnnToRLECases)
def test_annToRLE(coco_np: COCO, coco_pt: COCOpt, ann_id: int, result) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_np = coco_np.loadAnns(ann_id)
    ann_pt = coco_pt.loadAnns(ann_id)
    # get the RLE for the annotation
    rle_np = coco_np.annToRLE(ann_np[0])
    rle_pt = coco_pt.annToRLE(ann_pt[0])
    # compare the results
    assert rle_np == rle_pt
    assert rle_np["counts"] == rle_pt["counts"]
    assert rle_np["size"] == rle_pt["size"]
    assert rle_np["counts"] == result["counts"]
    assert rle_np["size"] == result["size"]
