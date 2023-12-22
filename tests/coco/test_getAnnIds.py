from pycocotools.coco import COCO as COCO
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class GetAnnIdsCases:
    def case_test(self) -> tuple:
        return (397133, 2096753)


@pytest.mark.benchmark(group="getAnnIds", warmup=True)
@parametrize_with_cases("img_id, result", cases=GetAnnIdsCases)
def test_getAnnIds_pt(benchmark, coco_pt: COCOpt, img_id: int, result: int) -> None:  # noqa: N802
    ann_ids_pt = benchmark(coco_pt.getAnnIds(imgIds=img_id))
    # compare the results
    assert ann_ids_pt == [result]


@pytest.mark.benchmark(group="getAnnIds", warmup=True)
@parametrize_with_cases("img_id, result", cases=GetAnnIdsCases)
def test_getAnnIds_np(benchmark, coco_np: COCO, img_id: int, result: int) -> None:  # noqa: N802
    ann_ids_np = benchmark(coco_np.getAnnIds(imgIds=img_id))
    # compare the results
    assert ann_ids_np == [result]


@parametrize_with_cases("img_id, result", cases=GetAnnIdsCases)
def test_getAnnIds(coco_np: COCO, coco_pt: COCOpt, img_id: int, result: int) -> None:  # noqa: N802
    # get the annotation ids for the image with id 397133
    ann_ids_np = coco_np.getAnnIds(imgIds=img_id)
    ann_ids_pt = coco_pt.getAnnIds(imgIds=img_id)
    # compare the results
    assert ann_ids_np == ann_ids_pt
    assert ann_ids_np == [result]
