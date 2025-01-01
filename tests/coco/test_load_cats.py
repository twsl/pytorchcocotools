from typing import cast

from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_cases import parametrize_with_cases

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811
from pytorchcocotools.internal.structure.categories import CocoCategoriesDetection


class LoadCatsCases:
    def case_test(self) -> tuple[int, list[dict]]:
        return (1, [{"supercategory": "person", "id": 1, "name": "person"}])


@pytest.mark.benchmark(group="loadCats", warmup=True)
@parametrize_with_cases("cat_ids, result", cases=LoadCatsCases)
def test_loadCats_pt(benchmark, coco_pt: COCOpt, cat_ids: int, result: list[dict]) -> None:  # noqa: N802
    # get the category ids for the image with id
    cat_pt = cast(list[CocoCategoriesDetection], benchmark(coco_pt.loadCats, cat_ids))
    # compare the results
    assert cat_pt[0].__dict__ == result[0]


@pytest.mark.benchmark(group="loadCats", warmup=True)
@parametrize_with_cases("cat_ids, result", cases=LoadCatsCases)
def test_loadCats_np(benchmark, coco_np: COCOnp, cat_ids: int, result: list[dict]) -> None:  # noqa: N802
    # get the category ids for the image with id
    cat_np = cast(list[dict], benchmark(coco_np.loadCats, cat_ids))
    # compare the results
    assert cat_np == result


@parametrize_with_cases("cat_ids, result", cases=LoadCatsCases)
def test_loadCats(coco_np: COCOnp, coco_pt: COCOpt, cat_ids: int, result: list[dict]) -> None:  # noqa: N802
    # get the category ids for the image with id 397133
    cat_np = coco_np.loadCats(cat_ids)
    cat_pt = coco_pt.loadCats(cat_ids)
    # compare the results
    for cnp, cpt in zip(cat_np, cat_pt, strict=False):
        assert cnp == cpt.__dict__
    for cnp, c in zip(cat_np, result, strict=False):
        assert cnp == c
