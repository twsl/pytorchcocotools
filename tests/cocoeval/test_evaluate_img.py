from typing import Any, TypeAlias, cast

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize, parametrize_with_cases
import torch

from pytorchcocotools.cocoeval import COCOeval as COCOevalpt
from pytorchcocotools.internal.cocoeval_types import EvalImgResult  # noqa: N811

TEST_DATA: TypeAlias = tuple[int, int, tuple[int, int], int, EvalImgResult]

RANGE1 = (0, int(1e5**2))


class BBoxCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return (
            1,
            1,
            RANGE1,
            1,
            EvalImgResult(
                dtScores=torch.tensor([0.5]),
                dtMatches=torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]),
            ),
        )

    def case_test_2(self) -> TEST_DATA:
        return (
            1,
            2,
            RANGE1,
            1,
            EvalImgResult(
                dtScores=torch.tensor([0.8]),
                dtMatches=torch.tensor([[2.0], [2.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
            ),
        )


class SegmCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return (
            1,
            1,
            RANGE1,
            1,
            EvalImgResult(
                dtScores=torch.tensor([0.5]),
                dtMatches=torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]),
            ),
        )

    def case_test_2(self) -> TEST_DATA:
        return (
            1,
            2,
            RANGE1,
            1,
            EvalImgResult(
                dtScores=torch.tensor([0.8]),
                dtMatches=torch.tensor([[2.0], [2.0], [2.0], [2.0], [2.0], [2.0], [2.0], [2.0], [0.0], [0.0]]),
            ),
        )


class KeypointCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return (
            1,
            4,
            RANGE1,
            1,
            EvalImgResult(
                dtScores=torch.tensor([0.5]),
                dtMatches=torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0]]),
            ),
        )


class COCOEvalCasesNp:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(
        self, eval_bbox_np: COCOevalnp, data: TEST_DATA
    ) -> tuple[COCOevalnp, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_bbox_np, img_id, cat_id, range, max_det, result)

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(
        self, eval_segm_np: COCOevalnp, data: TEST_DATA
    ) -> tuple[COCOevalnp, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_segm_np, img_id, cat_id, range, max_det, result)

    @parametrize_with_cases("data", cases=KeypointCases)
    def case_eval_keypoints(
        self, eval_keypoints_np: COCOevalnp, data: TEST_DATA
    ) -> tuple[COCOevalnp, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_keypoints_np, img_id, cat_id, range, max_det, result)


class COCOEvalCasesPt:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(
        self, eval_bbox_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_bbox_pt, img_id, cat_id, range, max_det, result)

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(
        self, eval_segm_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_segm_pt, img_id, cat_id, range, max_det, result)

    @parametrize_with_cases("data", cases=KeypointCases)
    def case_eval_keypoints(
        self, eval_keypoints_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_keypoints_pt, img_id, cat_id, range, max_det, result)


class COCOEvalCasesBoth:
    @parametrize_with_cases("data", cases=BBoxCases)
    def case_eval_bbox(
        self,
        eval_bbox_np: COCOevalnp,
        eval_bbox_pt: COCOevalpt,
        data: TEST_DATA,
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_bbox_np, eval_bbox_pt, img_id, cat_id, range, max_det, result)

    @parametrize_with_cases("data", cases=SegmCases)
    def case_eval_segm(
        self,
        eval_segm_np: COCOevalnp,
        eval_segm_pt: COCOevalpt,
        data: TEST_DATA,
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_segm_np, eval_segm_pt, img_id, cat_id, range, max_det, result)

    @parametrize_with_cases("data", cases=KeypointCases)
    def case_eval_keypoints(
        self,
        eval_keypoints_np: COCOevalnp,
        eval_keypoints_pt: COCOevalpt,
        data: TEST_DATA,
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_keypoints_np, eval_keypoints_pt, img_id, cat_id, range, max_det, result)


@pytest.mark.benchmark(group="evaluateImg", warmup=True)
@parametrize_with_cases("coco_eval_np, img_id, cat_id, range, max_det, result", cases=COCOEvalCasesNp)
def test_evaluateImg_np(  # noqa: N802
    benchmark: BenchmarkFixture,
    coco_eval_np: COCOevalnp,
    img_id: int,
    cat_id: int,
    range: tuple[int, int],
    max_det: int,
    result: EvalImgResult,
) -> None:
    coco_eval_np.evaluate()
    # result_np = coco_eval_np.evaluateImg(img_id, cat_id, list(range), max_det)
    result_np = cast(dict, benchmark(coco_eval_np.evaluateImg, img_id, cat_id, list(range), max_det))
    assert np.allclose(result_np["dtScores"], result.dtScores.tolist())
    assert np.allclose(result_np["dtMatches"], result.dtMatches.tolist())


@pytest.mark.benchmark(group="evaluateImg", warmup=True)
@parametrize_with_cases("coco_eval_pt, img_id, cat_id, range, max_det, result", cases=COCOEvalCasesPt)
def test_evaluateImg_pt(  # noqa: N802
    benchmark: BenchmarkFixture,
    coco_eval_pt: COCOevalpt,
    img_id: int,
    cat_id: int,
    range: tuple[int, int],
    max_det: int,
    result: EvalImgResult,
) -> None:
    coco_eval_pt.evaluate()
    # result_pt: EvalImgResult = coco_eval_pt.evaluateImg(img_id, cat_id, range, max_det)
    result_pt = cast(EvalImgResult, benchmark(coco_eval_pt.evaluateImg, img_id, cat_id, list(range), max_det))
    assert torch.allclose(result_pt.dtScores, result.dtScores)
    assert torch.allclose(result_pt.dtMatches, result.dtMatches)


@parametrize_with_cases("coco_eval_np, coco_eval_pt, img_id, cat_id, range, max_det, result", cases=COCOEvalCasesBoth)
def test_evaluateImg(  # noqa: N802
    coco_eval_np: COCOevalnp,
    coco_eval_pt: COCOevalpt,
    img_id: int,
    cat_id: int,
    range: tuple[int, int],
    max_det: int,
    result: EvalImgResult,
) -> None:
    coco_eval_np.evaluate()
    result_np = coco_eval_np.evaluateImg(img_id, cat_id, list(range), max_det)
    coco_eval_pt.evaluate()
    result_pt = coco_eval_pt.evaluateImg(img_id, cat_id, range, max_det)
    assert np.allclose(result_np["dtScores"], result.dtScores.tolist())
    assert np.allclose(result_np["dtMatches"], result.dtMatches.tolist())
    assert result_pt is not None
    assert torch.allclose(result_pt.dtScores, result.dtScores)
    assert torch.allclose(result_pt.dtMatches, result.dtMatches)
