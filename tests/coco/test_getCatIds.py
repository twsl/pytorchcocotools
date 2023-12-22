import numpy as np
from pycocotools.coco import COCO as COCO
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811  # noqa: N811
import torch


class GetCatIdsCases:
    def case_test(self) -> tuple:
        return (1, [1])


@pytest.mark.benchmark(group="getCatIds", warmup=True)
@parametrize_with_cases("cat_ids, result", cases=GetCatIdsCases)
def test_getCatIds_pt(benchmark, coco_pt: COCOpt, cat_ids, result) -> None:  # noqa: N802
    # get the category ids for the image with id
    cat_ids_pt = benchmark(coco_pt.getCatIds(catIds=cat_ids))
    # compare the results
    assert cat_ids_pt == result


@pytest.mark.benchmark(group="getCatIds", warmup=True)
@parametrize_with_cases("cat_ids, result", cases=GetCatIdsCases)
def test_getCatIds_np(benchmark, coco_np: COCO, cat_ids, result) -> None:  # noqa: N802
    # get the category ids for the image with id
    cat_ids_np = benchmark(coco_np.getCatIds(catIds=cat_ids))
    # compare the results
    assert cat_ids_np == result


@parametrize_with_cases("cat_ids, result", cases=GetCatIdsCases)
def test_getCatIds(coco_np: COCO, coco_pt: COCOpt, cat_ids, result) -> None:  # noqa: N802
    # get the category ids for the image with id 397133
    cat_ids_np = coco_np.getCatIds(catIds=cat_ids)
    cat_ids_pt = coco_pt.getCatIds(catIds=cat_ids)
    # compare the results
    assert cat_ids_np == cat_ids_pt
    assert cat_ids_np == result
