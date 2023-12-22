import numpy as np
from pycocotools.coco import COCO as COCO
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811  # noqa: N811  # noqa: N811
import torch


class GetCatIdsCases:
    def case_test(self) -> tuple:
        return (2096753, None)


@pytest.mark.benchmark(group="loadAnns", warmup=True)
@parametrize_with_cases("ann_ids, result", cases=GetCatIdsCases)
def test_loadAnns_pt(benchmark, coco_pt: COCOpt, ann_ids, result) -> None:  # noqa: N802
    # get the annotation ids for the id
    ann_pt = benchmark(coco_pt.loadAnns(ann_ids))
    # compare the results
    assert ann_pt == result


@pytest.mark.benchmark(group="loadAnns", warmup=True)
@parametrize_with_cases("ann_ids, result", cases=GetCatIdsCases)
def test_loadAnns_np(benchmark, coco_np: COCO, ann_ids, result) -> None:  # noqa: N802
    # get the annotation ids for the id
    ann_np = benchmark(coco_np.loadAnns(ann_ids))
    # compare the results
    assert ann_np == result


@parametrize_with_cases("ann_ids, result", cases=GetCatIdsCases)
def test_loadAnns(coco_np: COCO, coco_pt: COCOpt, ann_ids, result) -> None:  # noqa: N802
    # get the annotations for the id
    ann_np = coco_np.loadAnns(ann_ids)
    ann_pt = coco_pt.loadAnns(ann_ids)
    # compare the results
    assert ann_np == ann_pt
    assert ann_np == result
