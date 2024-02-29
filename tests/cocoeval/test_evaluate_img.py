from typing import Any

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_cases import parametrize, parametrize_with_cases
from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811
import torch

BBOX_DATA = [
    (1, 1, [[1.0]]),
    (1, 2, [[0.5625]]),
]
SEGM_DATA = [
    (1, 1, [[1.0]]),
    (1, 2, [[0.875]]),
]


class COCOEvalCasesNp:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(self, eval_bbox_np: COCOevalnp, data: tuple[int, int, Any]) -> tuple[COCOevalnp, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_bbox_np, img_id, cat_id, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(self, eval_segm_np: COCOevalnp, data: tuple[int, int, Any]) -> tuple[COCOevalnp, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_segm_np, img_id, cat_id, result)


class COCOEvalCasesPt:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(self, eval_bbox_pt: COCOevalpt, data: tuple[int, int, Any]) -> tuple[COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_bbox_pt, img_id, cat_id, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(self, eval_segm_pt: COCOevalpt, data: tuple[int, int, Any]) -> tuple[COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_segm_pt, img_id, cat_id, result)


class COCOEvalCasesBoth:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(
        self, eval_bbox_np: COCOevalnp, eval_bbox_pt: COCOevalpt, data: tuple[int, int, Any]
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_bbox_np, eval_bbox_pt, img_id, cat_id, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(
        self, eval_segm_np: COCOevalnp, eval_segm_pt: COCOevalpt, data: tuple[int, int, Any]
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_segm_np, eval_segm_pt, img_id, cat_id, result)


@pytest.mark.benchmark(group="evaluateImg", warmup=True)
@parametrize_with_cases("coco_eval_np, img_id, cat_id, result", cases=COCOEvalCasesNp)
def test_evaluateImg_np(benchmark, coco_eval_np: COCOevalnp, img_id: int, cat_id: int, result):  # noqa: N802
    coco_eval_np.evaluate()
    ious = coco_eval_np.evaluateImg(img_id, cat_id)
    # ious = benchmark(coco_eval_np.evaluateImg, img_id, cat_id)
    result = np.array(result)
    assert ious.shape == result.shape
    assert np.allclose(ious, result)


@pytest.mark.benchmark(group="evaluateImg", warmup=True)
@parametrize_with_cases("coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesPt)
def test_evaluateImg_pt(benchmark, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result):  # noqa: N802
    coco_eval_pt.evaluate()
    ious = coco_eval_pt.evaluateImg(img_id, cat_id)
    # ious = benchmark(coco_eval_pt.evaluateImg, img_id, cat_id)
    assert len(ious) == len(result)
    assert torch.allclose(ious, torch.Tensor(result))


@parametrize_with_cases("coco_eval_np, coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesBoth)
def test_evaluateImg(coco_eval_np: COCOevalnp, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result):  # noqa: N802
    coco_eval_np.evaluate()
    ious_np = coco_eval_np.evaluateImg(img_id, cat_id)
    coco_eval_pt.evaluate()
    ious_pt = coco_eval_pt.evaluateImg(img_id, cat_id)
    assert np.allclose(ious_np, ious_pt)
    assert torch.allclose(ious_pt, torch.Tensor(result))
