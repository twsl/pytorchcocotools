from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class LoadCatsCases:
    def case_test(self) -> tuple:
        return (1, [{"supercategory": "person", "id": 1, "name": "person"}])


@pytest.mark.benchmark(group="loadCats", warmup=True)
@parametrize_with_cases("cat_ids, result", cases=LoadCatsCases)
def test_loadCats_pt(benchmark, coco_pt: COCOpt, cat_ids, result) -> None:  # noqa: N802
    # get the category ids for the image with id
    cat_pt = benchmark(coco_pt.loadCats, cat_ids)
    # compare the results
    assert cat_pt == result


@pytest.mark.benchmark(group="loadCats", warmup=True)
@parametrize_with_cases("cat_ids, result", cases=LoadCatsCases)
def test_loadCats_np(benchmark, coco_np: COCOnp, cat_ids, result) -> None:  # noqa: N802
    # get the category ids for the image with id
    cat_np = benchmark(coco_np.loadCats, cat_ids)
    # compare the results
    assert cat_np == result


@parametrize_with_cases("cat_ids, result", cases=LoadCatsCases)
def test_loadCats(coco_np: COCOnp, coco_pt: COCOpt, cat_ids, result) -> None:  # noqa: N802
    # get the category ids for the image with id 397133
    cat_np = coco_np.loadCats(cat_ids)
    cat_pt = coco_pt.loadCats(cat_ids)
    # compare the results
    assert cat_np == cat_pt
    assert cat_np == result
