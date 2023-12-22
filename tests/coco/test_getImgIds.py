import numpy as np
from pycocotools.coco import COCO as COCO
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811  # noqa: N811  # noqa: N811
import torch


class GetImgIdsCases:
    def case_test(self) -> tuple:
        return (1, [397133])


@pytest.mark.benchmark(group="getImgIds", warmup=True)
@parametrize_with_cases("cat_ids, result", cases=GetImgIdsCases)
def test_getImgIds_pt(benchmark, coco_pt: COCOpt, cat_ids, result) -> None:  # noqa: N802
    # get the category ids for the image with id
    img_ids_pt = benchmark(coco_pt.getImgIds(catIds=cat_ids))
    # compare the results
    assert img_ids_pt == result


@pytest.mark.benchmark(group="getImgIds", warmup=True)
@parametrize_with_cases("cat_ids, result", cases=GetImgIdsCases)
def test_getImgIds_np(benchmark, coco_np: COCO, cat_ids, result) -> None:  # noqa: N802
    # get the category ids for the image with id
    img_ids_np = benchmark(coco_np.getImgIds(catIds=cat_ids))
    # compare the results
    assert img_ids_np == result


@parametrize_with_cases("cat_ids, result", cases=GetImgIdsCases)
def test_getCatIds(coco_np: COCO, coco_pt: COCOpt, cat_ids, result) -> None:  # noqa: N802
    # get the category ids for the image with id 397133
    img_ids_np = coco_np.getImgIds(catIds=cat_ids)
    img_ids_pt = coco_pt.getImgIds(catIds=cat_ids)
    # compare the results
    assert img_ids_np == img_ids_pt
    assert img_ids_np == result
