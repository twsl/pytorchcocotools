from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class GetCatIdsCases:
    def case_test(self) -> tuple:
        result = [
            {
                "segmentation": [
                    [
                        266.83,
                        189.37,
                        267.79,
                        175.29,
                        269.46,
                        170.04,
                        271.37,
                        165.98,
                        270.89,
                        163.12,
                        269.69,
                        159.54,
                        272.8,
                        156.44,
                        287.36,
                        156.44,
                        293.33,
                        157.87,
                        296.91,
                        160.49,
                        296.91,
                        161.21,
                        291.89,
                        161.92,
                        289.98,
                        165.03,
                        291.42,
                        169.56,
                        286.16,
                        196.54,
                    ],
                    [
                        266.35,
                        214.44,
                        270.41,
                        217.3,
                        276.38,
                        218.97,
                        282.11,
                        218.97,
                        285.93,
                        217.3,
                        286.88,
                        207.28,
                        267.07,
                        201.07,
                    ],
                ],
                "area": 1060.2075000000007,
                "iscrowd": 0,
                "image_id": 397133,
                "bbox": [266.35, 156.44, 30.56, 62.53],
                "category_id": 1,
                "id": 2096753,
            }
        ]

        return (2096753, result)


@pytest.mark.benchmark(group="loadAnns", warmup=True)
@parametrize_with_cases("ann_ids, result", cases=GetCatIdsCases)
def test_loadAnns_pt(benchmark, coco_pt: COCOpt, ann_ids, result) -> None:  # noqa: N802
    # get the annotation ids for the id
    ann_pt = benchmark(coco_pt.loadAnns, ann_ids)
    # compare the results
    assert ann_pt == result


@pytest.mark.benchmark(group="loadAnns", warmup=True)
@parametrize_with_cases("ann_ids, result", cases=GetCatIdsCases)
def test_loadAnns_np(benchmark, coco_np: COCOnp, ann_ids, result) -> None:  # noqa: N802
    # get the annotation ids for the id
    ann_np = benchmark(coco_np.loadAnns, ann_ids)
    # compare the results
    assert ann_np == result


@parametrize_with_cases("ann_ids, result", cases=GetCatIdsCases)
def test_loadAnns(coco_np: COCOnp, coco_pt: COCOpt, ann_ids, result) -> None:  # noqa: N802
    # get the annotations for the id
    ann_np = coco_np.loadAnns(ann_ids)
    ann_pt = coco_pt.loadAnns(ann_ids)
    # compare the results
    assert ann_np == ann_pt
    assert ann_np == result
