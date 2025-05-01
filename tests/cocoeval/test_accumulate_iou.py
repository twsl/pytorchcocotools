from typing import Any, TypeAlias, cast

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
from pycocotools.cocoeval import Params as Paramsnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize, parametrize_with_cases
import torch
from torch import Tensor

from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811
from pytorchcocotools.internal.cocoeval_types import Params as Paramspt

TEST_DATA: TypeAlias = tuple[Any, Tensor, Tensor, Tensor]


class BBoxCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return (
            None,
            torch.tensor(-0.5514851485148515),
            torch.tensor(-0.46245874587458746),
            torch.tensor(-0.4583333333333333),
        )


class SegmCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return (
            None,
            torch.tensor(-0.483003300330033),
            torch.tensor(-0.36666666666666664),
            torch.tensor(-0.36666666666666664),
        )


class COCOEvalCasesNp:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(self, eval_bbox_np: COCOevalnp, data: TEST_DATA) -> tuple[COCOevalnp, Any, tuple[Any, Any, Any]]:
        params, scores, precision, recall = data
        return (eval_bbox_np, params, (scores, precision, recall))

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(self, eval_segm_np: COCOevalnp, data: TEST_DATA) -> tuple[COCOevalnp, Any, tuple[Any, Any, Any]]:
        params, scores, precision, recall = data
        return (eval_segm_np, params, (scores, precision, recall))


class COCOEvalCasesPt:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(self, eval_bbox_pt: COCOevalpt, data: TEST_DATA) -> tuple[COCOevalpt, Any, tuple[Any, Any, Any]]:
        params, scores, precision, recall = data
        return (eval_bbox_pt, params, (scores, precision, recall))

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(self, eval_segm_pt: COCOevalpt, data: TEST_DATA) -> tuple[COCOevalpt, Any, tuple[Any, Any, Any]]:
        params, scores, precision, recall = data
        return (eval_segm_pt, params, (scores, precision, recall))


class COCOEvalCasesBoth:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(
        self, eval_bbox_np: COCOevalnp, eval_bbox_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalnp, COCOevalpt, Any, tuple[Any, Any, Any]]:
        params, scores, precision, recall = data
        return (eval_bbox_np, eval_bbox_pt, params, (scores, precision, recall))

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(
        self, eval_segm_np: COCOevalnp, eval_segm_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalnp, COCOevalpt, Any, tuple[Any, Any, Any]]:
        params, scores, precision, recall = data
        return (eval_segm_np, eval_segm_pt, params, (scores, precision, recall))


def convert_params(params: Paramspt | None) -> Paramsnp | None:
    if params is None:
        return None
    p = Paramsnp(params.iouType)
    p.imgIds = params.imgIds
    p.catIds = params.catIds
    p.areaRng = cast(list[list[float]], params.areaRng)
    p.maxDets = params.maxDets
    p.useCats = params.useCats
    p.iouThrs = params.iouThrs.numpy()
    p.recThrs = params.recThrs.numpy()
    p.kpt_oks_sigmas = params.kpt_oks_sigmas.numpy()
    p.areaRngLbl = cast(list[str], params.areaRngLbl)
    return p


@pytest.mark.benchmark(group="accumulate", warmup=True)
@parametrize_with_cases("coco_eval_np, params, result", cases=COCOEvalCasesNp)
def test_accumulate_np(
    benchmark: BenchmarkFixture,
    coco_eval_np: COCOevalnp,
    params: Paramspt | None,
    result: tuple[Tensor, Tensor, Tensor],
) -> None:
    params_np = convert_params(params)
    coco_eval_np.evaluate()
    # coco_eval_np.accumulate()
    benchmark(coco_eval_np.accumulate, params_np)
    scores, precision, recall = result
    assert np.allclose(coco_eval_np.eval["scores"].mean(), scores.numpy())
    assert np.allclose(coco_eval_np.eval["precision"].mean(), precision.numpy())
    assert np.allclose(coco_eval_np.eval["recall"].mean(), recall.numpy())


@pytest.mark.benchmark(group="accumulate", warmup=True)
@parametrize_with_cases("coco_eval_pt, params, result", cases=COCOEvalCasesPt)
def test_accumulate_pt(
    benchmark: BenchmarkFixture,
    coco_eval_pt: COCOevalpt,
    params: Paramspt | None,
    result: tuple[Tensor, Tensor, Tensor],
) -> None:
    coco_eval_pt.evaluate()
    coco_eval_pt.accumulate()
    # benchmark(coco_eval_pt.accumulate, params)
    scores, precision, recall = result
    assert torch.allclose(coco_eval_pt.eval.scores.mean(), scores)
    assert torch.allclose(coco_eval_pt.eval.precision.mean(), precision)
    assert torch.allclose(coco_eval_pt.eval.recall.mean(), recall)


@parametrize_with_cases("coco_eval_np, coco_eval_pt, params, result", cases=COCOEvalCasesBoth)
def test_accumulate(
    coco_eval_np: COCOevalnp, coco_eval_pt: COCOevalpt, params: Paramspt | None, result: tuple[Tensor, Tensor, Tensor]
) -> None:
    params_np = convert_params(params)
    coco_eval_np.evaluate()
    coco_eval_pt.evaluate()
    coco_eval_np.accumulate(params_np)
    coco_eval_pt.accumulate(params)
    scores, precision, recall = result
    assert torch.allclose(coco_eval_pt.eval.scores.mean(), scores)
    assert torch.allclose(coco_eval_pt.eval.precision.mean(), precision)
    assert torch.allclose(coco_eval_pt.eval.recall.mean(), recall)
    assert np.allclose(coco_eval_np.eval["scores"].mean(), scores.numpy())
    assert np.allclose(coco_eval_np.eval["precision"].mean(), precision.numpy())
    assert np.allclose(coco_eval_np.eval["recall"].mean(), recall.numpy())
