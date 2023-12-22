import numpy as np
from pycocotools.coco import COCO as COCO
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811  # noqa: N811  # noqa: N811  # noqa: N811
import torch


class LoadImgsCases:
    def case_test(self) -> tuple:
        return (
            397133,
            [
                {
                    "license": 1,
                    "file_name": "000000397133.jpg",
                    "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
                    "height": 427,
                    "width": 640,
                    "date_captured": "2013-11-14 17:02:52",
                    "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
                    "id": 397133,
                }
            ],
        )


@pytest.mark.benchmark(group="loadImgs", warmup=True)
@parametrize_with_cases("img_ids, result", cases=LoadImgsCases)
def test_loadImgs_pt(benchmark, coco_pt: COCOpt, img_ids, result) -> None:  # noqa: N802
    # get the image with id
    cat_pt = benchmark(coco_pt.loadImgs(img_ids))
    # compare the results
    assert cat_pt == result


@pytest.mark.benchmark(group="loadImgs", warmup=True)
@parametrize_with_cases("img_ids, result", cases=LoadImgsCases)
def test_loadImgs_np(benchmark, coco_np: COCO, img_ids, result) -> None:  # noqa: N802
    # get the image with id
    cat_np = benchmark(coco_np.loadImgs(img_ids))
    # compare the results
    assert cat_np == result


@parametrize_with_cases("img_ids, result", cases=LoadImgsCases)
def test_loadImgs(coco_np: COCO, coco_pt: COCOpt, img_ids, result) -> None:  # noqa: N802
    # get the image with id
    imgs_np = coco_np.loadImgs(img_ids)
    imgs_pt = coco_pt.loadImgs(img_ids)
    # compare the results
    assert imgs_np == imgs_pt
    assert imgs_np == result
