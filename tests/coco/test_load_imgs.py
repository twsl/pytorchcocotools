from typing import cast

from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811
from pytorchcocotools.internal.structure.images import CocoImage


class LoadImgsCases:
    def case_test_1(self) -> tuple[int | list[int], list[dict]]:
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
def test_loadImgs_pt(  # noqa: N802
    benchmark: BenchmarkFixture, coco_pt: COCOpt, img_ids: int | list[int], result: list[dict]
) -> None:
    # get the image with id
    imgs_pt = cast(list[CocoImage], benchmark(coco_pt.loadImgs, img_ids))
    # compare the results
    # assert cat_pt == result
    for img_np, img_pt in zip(result, imgs_pt, strict=False):
        assert img_np == img_pt.__dict__


@pytest.mark.benchmark(group="loadImgs", warmup=True)
@parametrize_with_cases("img_ids, result", cases=LoadImgsCases)
def test_loadImgs_np(  # noqa: N802
    benchmark: BenchmarkFixture, coco_np: COCOnp, img_ids: int | list[int], result: list[dict]
) -> None:
    # get the image with id
    imgs_np = cast(list[dict], benchmark(coco_np.loadImgs, img_ids))
    # compare the results
    assert imgs_np == result


@parametrize_with_cases("img_ids, result", cases=LoadImgsCases)
def test_loadImgs(coco_np: COCOnp, coco_pt: COCOpt, img_ids: int | list[int], result: list[dict]) -> None:  # noqa: N802
    # get the image with id
    imgs_np = coco_np.loadImgs(img_ids)
    imgs_pt = coco_pt.loadImgs(img_ids)
    # compare the results
    for img_np, img_pt in zip(imgs_np, imgs_pt, strict=False):
        assert img_np == img_pt.__dict__
    for img_np, img in zip(imgs_np, result, strict=False):
        assert img_np == img
