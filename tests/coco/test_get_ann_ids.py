from typing import cast

from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_cases import parametrize_with_cases

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class GetAnnIdsCases:
    def case_test(self) -> tuple[int | list[int], int | list[int], float | list[float], list[int]]:
        return (397133, [], [], [2096753])


@pytest.mark.benchmark(group="getAnnIds", warmup=True)
@parametrize_with_cases("img_id, cat_ids, area_rng, result", cases=GetAnnIdsCases)
def test_getAnnIds_pt(  # noqa: N802
    benchmark,
    coco_pt: COCOpt,
    img_id: int | list[int],
    cat_ids: int | list[int],
    area_rng: float | list[float],
    result: list[int],
) -> None:
    ann_ids_pt = cast(list[int], benchmark(coco_pt.getAnnIds, img_id, cat_ids, area_rng))
    # compare the results
    assert ann_ids_pt == result


@pytest.mark.benchmark(group="getAnnIds", warmup=True)
@parametrize_with_cases("img_id, cat_ids, area_rng, result", cases=GetAnnIdsCases)
def test_getAnnIds_np(  # noqa: N802
    benchmark,
    coco_np: COCOnp,
    img_id: int | list[int],
    cat_ids: int | list[int],
    area_rng: float | list[float],
    result: list[int],
) -> None:
    ann_ids_np = cast(list[int], benchmark(coco_np.getAnnIds, img_id, cat_ids, area_rng))
    # compare the results
    assert ann_ids_np == result


@parametrize_with_cases("img_id, cat_ids, area_rng, result", cases=GetAnnIdsCases)
def test_getAnnIds(  # noqa: N802
    coco_np: COCOnp,
    coco_pt: COCOpt,
    img_id: int | list[int],
    cat_ids: int | list[int],
    area_rng: float | list[float],
    result: list[int],
) -> None:
    # get the annotation ids for the image with id
    ann_ids_np = coco_np.getAnnIds(img_id, cat_ids, area_rng)  # pyright: ignore[reportArgumentType]
    ann_ids_pt = coco_pt.getAnnIds(img_id, cat_ids, area_rng)
    # compare the results
    assert ann_ids_np == ann_ids_pt
    assert ann_ids_np == result
