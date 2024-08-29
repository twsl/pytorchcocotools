from typing import Any

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_cases import parametrize, parametrize_with_cases
import torch

from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811

RANGE1 = (0, int(1e5**2))

BBOX_DATA = [
    (1, 1, RANGE1, 1, [[1.0]]),
    (1, 2, RANGE1, 1, [[0.5625]]),
]
SEGM_DATA = [
    (1, 1, RANGE1, 1, [[1.0]]),
    (1, 2, RANGE1, 1, [[0.875]]),
]

KEYPOINTS_DATA = [
    (1, 1, RANGE1, 1, [[1.0]]),
]


class COCOEvalCasesNp:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(
        self, eval_bbox_np: COCOevalnp, data: tuple[int, int, tuple[int, int], int, Any]
    ) -> tuple[COCOevalnp, int, int, tuple[int, int], int, Any]:
        img_id, cat_id, range, max_det, result = data
        return (eval_bbox_np, img_id, cat_id, range, max_det, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(
        self, eval_segm_np: COCOevalnp, data: tuple[int, int, tuple[int, int], int, Any]
    ) -> tuple[COCOevalnp, int, int, tuple[int, int], int, Any]:
        img_id, cat_id, range, max_det, result = data
        return (eval_segm_np, img_id, cat_id, range, max_det, result)

    @parametrize(data=KEYPOINTS_DATA)
    def case_eval_keypoints(
        self, eval_keypoints_np: COCOevalnp, data: tuple[int, int, tuple[int, int], int, Any]
    ) -> tuple[COCOevalnp, int, int, tuple[int, int], int, Any]:
        img_id, cat_id, range, max_det, result = data
        return (eval_keypoints_np, img_id, cat_id, range, max_det, result)


class COCOEvalCasesPt:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(
        self, eval_bbox_pt: COCOevalpt, data: tuple[int, int, tuple[int, int], int, Any]
    ) -> tuple[COCOevalpt, int, int, tuple[int, int], int, Any]:
        img_id, cat_id, range, max_det, result = data
        return (eval_bbox_pt, img_id, cat_id, range, max_det, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(
        self, eval_segm_pt: COCOevalpt, data: tuple[int, int, tuple[int, int], int, Any]
    ) -> tuple[COCOevalpt, int, int, tuple[int, int], int, Any]:
        img_id, cat_id, range, max_det, result = data
        return (eval_segm_pt, img_id, cat_id, range, max_det, result)

    @parametrize(data=KEYPOINTS_DATA)
    def case_eval_keypoints(
        self, eval_keypoints_pt: COCOevalpt, data: tuple[int, int, tuple[int, int], int, Any]
    ) -> tuple[COCOevalpt, int, int, tuple[int, int], int, Any]:
        img_id, cat_id, range, max_det, result = data
        return (eval_keypoints_pt, img_id, cat_id, range, max_det, result)


class COCOEvalCasesBoth:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(
        self, eval_bbox_np: COCOevalnp, eval_bbox_pt: COCOevalpt, data: tuple[int, int, tuple[int, int], int, Any]
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, tuple[int, int], int, Any]:
        img_id, cat_id, range, max_det, result = data
        return (eval_bbox_np, eval_bbox_pt, img_id, cat_id, range, max_det, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(
        self, eval_segm_np: COCOevalnp, eval_segm_pt: COCOevalpt, data: tuple[int, int, tuple[int, int], int, Any]
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, tuple[int, int], int, Any]:
        img_id, cat_id, range, max_det, result = data
        return (eval_segm_np, eval_segm_pt, img_id, cat_id, range, max_det, result)

    @parametrize(data=KEYPOINTS_DATA)
    def case_eval_keypoints(
        self,
        eval_keypoints_np: COCOevalnp,
        eval_keypoints_pt: COCOevalpt,
        data: tuple[int, int, tuple[int, int], int, Any],
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, tuple[int, int], int, Any]:
        img_id, cat_id, range, max_det, result = data
        return (eval_keypoints_np, eval_keypoints_pt, img_id, cat_id, range, max_det, result)


@pytest.mark.benchmark(group="evaluateImg", warmup=True)
@parametrize_with_cases("coco_eval_np, img_id, cat_id, result", cases=COCOEvalCasesNp)
def test_evaluateImg_np(  # noqa: N802
    benchmark, coco_eval_np: COCOevalnp, img_id: int, cat_id: int, range: tuple[int, int], max_det: int, result
) -> None:
    coco_eval_np.evaluate()
    result_np = coco_eval_np.evaluateImg(img_id, cat_id, list(range), max_det)
    # ious = benchmark(coco_eval_np.evaluateImg, img_id, cat_id)
    assert result_np == result.__dict__


@pytest.mark.benchmark(group="evaluateImg", warmup=True)
@parametrize_with_cases("coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesPt)
def test_evaluateImg_pt(  # noqa: N802
    benchmark, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, range: tuple[int, int], max_det: int, result
) -> None:
    coco_eval_pt.evaluate()
    result_pt = coco_eval_pt.evaluateImg(img_id, cat_id, range, max_det)
    # ious = benchmark(coco_eval_pt.evaluateImg, img_id, cat_id)
    assert result_pt == result


@parametrize_with_cases("coco_eval_np, coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesBoth)
def test_evaluateImg(  # noqa: N802
    coco_eval_np: COCOevalnp,
    coco_eval_pt: COCOevalpt,
    img_id: int,
    cat_id: int,
    range: tuple[int, int],
    max_det: int,
    result,
) -> None:
    coco_eval_np.evaluate()
    result_np = coco_eval_np.evaluateImg(img_id, cat_id, list(range), max_det)
    coco_eval_pt.evaluate()
    result_pt = coco_eval_pt.evaluateImg(img_id, cat_id, range, max_det)
    assert result_np == result_pt.__dict__
    assert result_pt == result
