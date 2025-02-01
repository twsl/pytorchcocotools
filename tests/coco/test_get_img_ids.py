from typing import cast

from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class GetImgIdsCases:
    def case_test_itself(self) -> tuple[int | list[int], int | list[int], list[int]]:
        return (397133, 1, [397133])

    def case_test_all_cats(self) -> tuple[int | list[int], int | list[int], list[int]]:
        return ([], 1, [80932, 397133])


@pytest.mark.benchmark(group="getImgIds", warmup=True)
@parametrize_with_cases("img_ids, cat_ids, result", cases=GetImgIdsCases)
def test_getImgIds_pt(  # noqa: N802
    benchmark: BenchmarkFixture, coco_pt: COCOpt, img_ids: int | list[int], cat_ids: int | list[int], result: list[int]
) -> None:
    # get the category ids for the image with id
    img_ids_pt = cast(list[int], benchmark(coco_pt.getImgIds, img_ids, cat_ids))
    # compare the results
    assert img_ids_pt == result


@pytest.mark.benchmark(group="getImgIds", warmup=True)
@parametrize_with_cases("img_ids, cat_ids, result", cases=GetImgIdsCases)
def test_getImgIds_np(  # noqa: N802
    benchmark: BenchmarkFixture, coco_np: COCOnp, img_ids: int | list[int], cat_ids: int | list[int], result: list[int]
) -> None:
    # get the category ids for the image with id
    img_ids_np = cast(list[int], benchmark(coco_np.getImgIds, img_ids, cat_ids))
    # compare the results
    assert img_ids_np == result


@parametrize_with_cases("img_ids, cat_ids, result", cases=GetImgIdsCases)
def test_getCatIds(  # noqa: N802
    coco_np: COCOnp, coco_pt: COCOpt, img_ids: int | list[int], cat_ids: int | list[int], result: list[int]
) -> None:
    # get the category ids for the image with id 397133
    img_ids_np = coco_np.getImgIds(img_ids, cat_ids)
    img_ids_pt = coco_pt.getImgIds(img_ids, cat_ids)
    # compare the results
    assert img_ids_np == img_ids_pt
    assert img_ids_np == result
