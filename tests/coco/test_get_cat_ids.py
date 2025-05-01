from typing import cast

from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class GetCatIdsCases:
    def case_test_1(self) -> tuple[str | list[str], str | list[str], int | list[int], list[int]]:
        return ([], [], 1, [1])


@pytest.mark.benchmark(group="getCatIds", warmup=True)
@parametrize_with_cases("cat_nms, sup_nms, cat_ids, result", cases=GetCatIdsCases)
def test_getCatIds_pt(  # noqa: N802
    benchmark: BenchmarkFixture,
    coco_pt: COCOpt,
    cat_nms: str | list[str],
    sup_nms: str | list[str],
    cat_ids: int | list[int],
    result,
) -> None:
    # get the category ids for the image with id
    cat_ids_pt = cast(list[int], benchmark(coco_pt.getCatIds, cat_nms, sup_nms, cat_ids))
    # compare the results
    assert cat_ids_pt == result


@pytest.mark.benchmark(group="getCatIds", warmup=True)
@parametrize_with_cases("cat_nms, sup_nms, cat_ids, result", cases=GetCatIdsCases)
def test_getCatIds_np(  # noqa: N802
    benchmark: BenchmarkFixture,
    coco_np: COCOnp,
    cat_nms: str | list[str],
    sup_nms: str | list[str],
    cat_ids: int | list[int],
    result,
) -> None:
    # get the category ids for the image with id
    cat_ids_np = cast(list[int], benchmark(coco_np.getCatIds, cat_nms, sup_nms, cat_ids))
    # compare the results
    assert cat_ids_np == result


@parametrize_with_cases("cat_nms, sup_nms, cat_ids, result", cases=GetCatIdsCases)
def test_getCatIds(  # noqa: N802
    coco_np: COCOnp,
    coco_pt: COCOpt,
    cat_nms: str | list[str],
    sup_nms: str | list[str],
    cat_ids: int | list[int],
    result,
) -> None:
    # get the category ids for the image with id
    cat_ids_np = coco_np.getCatIds(cat_nms, sup_nms, cat_ids)
    cat_ids_pt = coco_pt.getCatIds(cat_nms, sup_nms, cat_ids)
    # compare the results
    assert cat_ids_np == cat_ids_pt
    assert cat_ids_np == result
