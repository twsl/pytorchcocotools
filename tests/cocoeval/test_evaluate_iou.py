from typing import Any, TypeAlias, cast

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize, parametrize_with_cases
import torch
from torch import Tensor

from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811

TEST_DATA: TypeAlias = dict[tuple[int, int], list[list[float]]]


# TODO: find better results from evaluate to compare against
class BBoxCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return {
            (1, 1): [[1.0]],
            (1, 2): [[0.5625]],
            (2, 2): [[0.64]],
        }


class SegmCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return {
            (1, 1): [[1.0]],
            (1, 2): [[0.875]],
            (2, 2): [[0.85790885]],
        }


class COCOEvalCasesNp:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(self, eval_bbox_np: COCOevalnp, data: TEST_DATA) -> tuple[COCOevalnp, Any]:
        result = data
        return (eval_bbox_np, result)

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(self, eval_segm_np: COCOevalnp, data: TEST_DATA) -> tuple[COCOevalnp, Any]:
        result = data
        return (eval_segm_np, result)


class COCOEvalCasesPt:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(self, eval_bbox_pt: COCOevalpt, data: TEST_DATA) -> tuple[COCOevalpt, Any]:
        result = data
        return (eval_bbox_pt, result)

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(self, eval_segm_pt: COCOevalpt, data: TEST_DATA) -> tuple[COCOevalpt, Any]:
        result = data
        return (eval_segm_pt, result)


class COCOEvalCasesBoth:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(
        self, eval_bbox_np: COCOevalnp, eval_bbox_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalnp, COCOevalpt, Any]:
        result = data
        return (eval_bbox_np, eval_bbox_pt, result)

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(
        self, eval_segm_np: COCOevalnp, eval_segm_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalnp, COCOevalpt, Any]:
        result = data
        return (eval_segm_np, eval_segm_pt, result)


@pytest.mark.benchmark(group="evaluateIoU", warmup=True)
@parametrize_with_cases("coco_eval_np, result", cases=COCOEvalCasesNp)
def test_evaluate_np(benchmark: BenchmarkFixture, coco_eval_np: COCOevalnp, result: TEST_DATA) -> None:  # noqa: N802
    # coco_eval_np.evaluate()
    benchmark(coco_eval_np.evaluate)
    for combo in result:
        iou = coco_eval_np.ious[combo]
        assert iou is not None, f"IOU np is None for combo {combo}"
        assert np.allclose(iou, np.array(result[combo])), f"IOU np mismatch for combo {combo}"


@pytest.mark.benchmark(group="evaluateIoU", warmup=True)
@parametrize_with_cases("coco_eval_pt, result", cases=COCOEvalCasesPt)
def test_evaluate_pt(benchmark: BenchmarkFixture, coco_eval_pt: COCOevalpt, result: TEST_DATA) -> None:  # noqa: N802
    # coco_eval_pt.evaluate()
    benchmark(coco_eval_pt.evaluate)
    for combo in result:
        iou = coco_eval_pt.ious[combo]
        assert iou is not None, f"IOU pt is None for combo {combo}"
        assert torch.allclose(iou, torch.tensor(result[combo])), f"IOU pt mismatch for combo {combo}"


@parametrize_with_cases("coco_eval_np, coco_eval_pt, result", cases=COCOEvalCasesBoth)
def test_evaluate(coco_eval_np: COCOevalnp, coco_eval_pt: COCOevalpt, result: TEST_DATA) -> None:  # noqa: N802
    coco_eval_np.evaluate()
    coco_eval_pt.evaluate()
    for combo in result:
        iou_np = coco_eval_np.ious[combo]
        iou_pt = coco_eval_pt.ious[combo]
        assert iou_np is not None, f"IOU np is None for combo {combo}"
        assert np.allclose(iou_np, np.array(result[combo])), f"IOU np mismatch for combo {combo}"
        assert iou_pt is not None, f"IOU pt is None for combo {combo}"
        assert torch.allclose(iou_pt, torch.tensor(result[combo])), f"IOU pt mismatch for combo {combo}"
