from typing import Any, TypeAlias, cast

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize, parametrize_with_cases
import torch
from torch import Tensor

from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811

TEST_DATA: TypeAlias = tuple[int, int, Any]


class BBoxCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return (1, 1, [[1.0]])

    def case_test_2(self) -> TEST_DATA:
        return (1, 2, [[0.5625]])


class SegmCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return (1, 1, [[1.0]])

    def case_test_2(self) -> TEST_DATA:
        return (1, 2, [[0.875]])


class InputsBBoxCases:
    def case_img101_cat4(self) -> TEST_DATA:
        # 1 DT x 1 GT
        return (101, 4, [[0.7756590016825576]])

    def case_img102_cat2(self) -> TEST_DATA:
        # 1 DT x 2 GT
        return (102, 2, [[0.20396039628170556, 0.9213161659513592]])

    def case_img103_cat0(self) -> TEST_DATA:
        # 5 DT x 5 GT
        return (
            103,
            0,
            [
                [0.0, 0.0, 0.0, 0.0, 0.8775260257195345],
                [0.0, 0.0, 0.0, 0.8811645870469401, 0.0],
                [0.0, 0.7427652733118973, 0.0, 0.0, 0.0],
                [0.8970133882595265, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
        )

    def case_img103_cat4(self) -> TEST_DATA:
        # 1 DT x 1 GT
        return (103, 4, [[0.8387196824018361]])

    def case_img104_cat49(self) -> TEST_DATA:
        # 9 DT x 10 GT
        return (
            104,
            49,
            [
                [
                    0.07665926856885018,
                    0.0,
                    0.041623054912310546,
                    0.037918442010694726,
                    1.0,
                    0.009129903074799451,
                    0.08187583421637974,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.31322917613407897,
                    0.0,
                    0.8911860718171924,
                    0.05513067484117222,
                    0.037537390222923205,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.11218717858007658,
                    0.27786272061290224,
                    0.0,
                    0.4202534690104182,
                    0.0,
                    0.0,
                    0.0,
                    0.8809523809523808,
                    0.06344782346547402,
                    0.010937682062505413,
                ],
                [
                    1.0,
                    0.0,
                    0.36085596875187714,
                    0.4166955416311106,
                    0.07665926856885018,
                    0.0,
                    0.0,
                    0.1425617713480852,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.12272257855784072,
                    0.0,
                    0.0058746312043412175,
                    0.0,
                    0.01082829278266281,
                    0.0,
                    0.004449679361340149,
                    0.12376536515616296,
                    0.6093750000000001,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.08187583421637974, 0.026643481336300255, 0.856218547807333, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.617004978935274,
                    0.0,
                    0.057863609661599806,
                    0.0,
                    0.0016671355661037443,
                    0.0,
                    0.09431915986161085,
                    0.8089209038203887,
                    0.1203864692970297,
                ],
                [0.0, 0.34606765619059543, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7370727432077125, 0.041260863851785325],
                [
                    0.23639760018553962,
                    0.3640540490960734,
                    0.0,
                    0.6719967199671996,
                    0.020828713804405763,
                    0.0,
                    0.0,
                    0.5497176952658371,
                    0.10694935529422282,
                    0.08934648587749848,
                ],
            ],
        )


class COCOEvalCasesNp:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(self, eval_bbox_np: COCOevalnp, data: TEST_DATA) -> tuple[COCOevalnp, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_bbox_np, img_id, cat_id, result)

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(self, eval_segm_np: COCOevalnp, data: TEST_DATA) -> tuple[COCOevalnp, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_segm_np, img_id, cat_id, result)

    @parametrize_with_cases("data", cases=InputsBBoxCases)
    def case_eval_inputs_bbox(
        self, eval_bbox_inputs_np: COCOevalnp, data: TEST_DATA
    ) -> tuple[COCOevalnp, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_bbox_inputs_np, img_id, cat_id, result)


class COCOEvalCasesPt:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(self, eval_bbox_pt: COCOevalpt, data: TEST_DATA) -> tuple[COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_bbox_pt, img_id, cat_id, result)

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(self, eval_segm_pt: COCOevalpt, data: TEST_DATA) -> tuple[COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_segm_pt, img_id, cat_id, result)

    @parametrize_with_cases("data", cases=InputsBBoxCases)
    def case_eval_inputs_bbox(
        self, eval_bbox_inputs_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_bbox_inputs_pt, img_id, cat_id, result)


class COCOEvalCasesBoth:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(
        self, eval_bbox_np: COCOevalnp, eval_bbox_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_bbox_np, eval_bbox_pt, img_id, cat_id, result)

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(
        self, eval_segm_np: COCOevalnp, eval_segm_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_segm_np, eval_segm_pt, img_id, cat_id, result)

    @parametrize_with_cases("data", cases=InputsBBoxCases)
    def case_eval_inputs_bbox(
        self, eval_bbox_inputs_np: COCOevalnp, eval_bbox_inputs_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_bbox_inputs_np, eval_bbox_inputs_pt, img_id, cat_id, result)


@pytest.mark.benchmark(group="computeIoU", warmup=True)
@parametrize_with_cases("coco_eval_np, img_id, cat_id, result", cases=COCOEvalCasesNp)
def test_computeIoU_np(  # noqa: N802
    benchmark: BenchmarkFixture, coco_eval_np: COCOevalnp, img_id: int, cat_id: int, result: Any
) -> None:
    # ious = coco_eval_np.computeIoU(img_id, cat_id)
    ious = cast(np.ndarray, benchmark(coco_eval_np.computeIoU, img_id, cat_id))
    result = np.array(result)
    assert ious.shape == result.shape
    assert np.allclose(ious, result)


@pytest.mark.benchmark(group="computeIoU", warmup=True)
@parametrize_with_cases("coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesPt)
def test_computeIoU_pt(  # noqa: N802
    benchmark: BenchmarkFixture, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result: Any
) -> None:
    # ious = coco_eval_pt.computeIoU(img_id, cat_id)
    ious = cast(Tensor, benchmark(coco_eval_pt.computeIoU, img_id, cat_id))
    result = torch.tensor(result, dtype=torch.float32)
    assert ious.shape == result.shape
    assert torch.allclose(ious, result)


@parametrize_with_cases("coco_eval_np, coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesBoth)
def test_computeIoU(coco_eval_np: COCOevalnp, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result: Any) -> None:  # noqa: N802
    ious_np = coco_eval_np.computeIoU(img_id, cat_id)
    ious_pt = coco_eval_pt.computeIoU(img_id, cat_id)
    assert np.allclose(ious_np, np.array(result))
    assert torch.allclose(ious_pt, torch.tensor(result, dtype=torch.float32))
