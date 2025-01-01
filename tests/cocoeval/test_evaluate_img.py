from typing import Any

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_cases import parametrize, parametrize_with_cases
import torch

from pytorchcocotools.cocoeval import COCOeval as COCOevalpt
from pytorchcocotools.internal.cocoeval_types import EvalImgResult  # noqa: N811

RANGE1 = (0, int(1e5**2))

BBOX_DATA = [
    (1, 1, RANGE1, 1, EvalImgResult(dtScores=torch.tensor([0.5]))),
    (1, 2, RANGE1, 1, EvalImgResult(dtScores=torch.tensor([0.8]))),
]
SEGM_DATA = [
    (1, 1, RANGE1, 1, EvalImgResult(dtScores=torch.tensor([0.5]))),
    (1, 2, RANGE1, 1, EvalImgResult(dtScores=torch.tensor([0.8]))),
]

KEYPOINTS_DATA = [
    (1, 4, RANGE1, 1, EvalImgResult(dtScores=torch.tensor([0.5]))),
]


class COCOEvalCasesNp:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(
        self, eval_bbox_np: COCOevalnp, data: tuple[int, int, tuple[int, int], int, EvalImgResult]
    ) -> tuple[COCOevalnp, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_bbox_np, img_id, cat_id, range, max_det, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(
        self, eval_segm_np: COCOevalnp, data: tuple[int, int, tuple[int, int], int, EvalImgResult]
    ) -> tuple[COCOevalnp, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_segm_np, img_id, cat_id, range, max_det, result)

    @parametrize(data=KEYPOINTS_DATA)
    def case_eval_keypoints(
        self, eval_keypoints_np: COCOevalnp, data: tuple[int, int, tuple[int, int], int, EvalImgResult]
    ) -> tuple[COCOevalnp, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_keypoints_np, img_id, cat_id, range, max_det, result)


class COCOEvalCasesPt:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(
        self, eval_bbox_pt: COCOevalpt, data: tuple[int, int, tuple[int, int], int, EvalImgResult]
    ) -> tuple[COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_bbox_pt, img_id, cat_id, range, max_det, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(
        self, eval_segm_pt: COCOevalpt, data: tuple[int, int, tuple[int, int], int, EvalImgResult]
    ) -> tuple[COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_segm_pt, img_id, cat_id, range, max_det, result)

    @parametrize(data=KEYPOINTS_DATA)
    def case_eval_keypoints(
        self, eval_keypoints_pt: COCOevalpt, data: tuple[int, int, tuple[int, int], int, EvalImgResult]
    ) -> tuple[COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_keypoints_pt, img_id, cat_id, range, max_det, result)


class COCOEvalCasesBoth:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(
        self,
        eval_bbox_np: COCOevalnp,
        eval_bbox_pt: COCOevalpt,
        data: tuple[int, int, tuple[int, int], int, EvalImgResult],
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_bbox_np, eval_bbox_pt, img_id, cat_id, range, max_det, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(
        self,
        eval_segm_np: COCOevalnp,
        eval_segm_pt: COCOevalpt,
        data: tuple[int, int, tuple[int, int], int, EvalImgResult],
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_segm_np, eval_segm_pt, img_id, cat_id, range, max_det, result)

    @parametrize(data=KEYPOINTS_DATA)
    def case_eval_keypoints(
        self,
        eval_keypoints_np: COCOevalnp,
        eval_keypoints_pt: COCOevalpt,
        data: tuple[int, int, tuple[int, int], int, EvalImgResult],
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, tuple[int, int], int, EvalImgResult]:
        img_id, cat_id, range, max_det, result = data
        return (eval_keypoints_np, eval_keypoints_pt, img_id, cat_id, range, max_det, result)


@pytest.mark.benchmark(group="evaluateImg", warmup=True)
@parametrize_with_cases("coco_eval_np, img_id, cat_id, range, max_det, result", cases=COCOEvalCasesNp)
def test_evaluateImg_np(  # noqa: N802
    benchmark,
    coco_eval_np: COCOevalnp,
    img_id: int,
    cat_id: int,
    range: tuple[int, int],
    max_det: int,
    result: EvalImgResult,
) -> None:
    coco_eval_np.evaluate()
    result_np = coco_eval_np.evaluateImg(img_id, cat_id, list(range), max_det)
    # result_np = cast(dict, benchmark(coco_eval_np.evaluateImg, img_id, cat_id, list(range), max_det))
    assert np.allclose(result_np["dtScores"], result.dtScores.tolist())
    # assert result_np == result.__dict__


@pytest.mark.benchmark(group="evaluateImg", warmup=True)
@parametrize_with_cases("coco_eval_pt, img_id, cat_id, range, max_det, result", cases=COCOEvalCasesPt)
def test_evaluateImg_pt(  # noqa: N802
    benchmark,
    coco_eval_pt: COCOevalpt,
    img_id: int,
    cat_id: int,
    range: tuple[int, int],
    max_det: int,
    result: EvalImgResult,
) -> None:
    coco_eval_pt.evaluate()
    result_pt: EvalImgResult = coco_eval_pt.evaluateImg(img_id, cat_id, range, max_det)
    # result_pt = cast(EvalImgResult, benchmark(coco_eval_pt.evaluateImg, img_id, cat_id, list(range), max_det))
    assert torch.allclose(result_pt.dtScores, result.dtScores)


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
    assert torch.allclose(result_pt.dtScores, result.dtScores)
