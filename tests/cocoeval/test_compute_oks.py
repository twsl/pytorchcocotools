from typing import Any

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_cases import parametrize, parametrize_with_cases
from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811
import torch

KEYPOINTS_DATA = [
    (1, 1, [[1.0]]),
]


class COCOEvalCasesNp:
    @parametrize(data=KEYPOINTS_DATA)
    def case_eval_keypoints(
        self, eval_keypoints_np: COCOevalnp, data: tuple[int, int, Any]
    ) -> tuple[COCOevalnp, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_keypoints_np, img_id, cat_id, result)


class COCOEvalCasesPt:
    @parametrize(data=KEYPOINTS_DATA)
    def case_eval_keypoints(
        self, eval_keypoints_pt: COCOevalpt, data: tuple[int, int, Any]
    ) -> tuple[COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_keypoints_pt, img_id, cat_id, result)


class COCOEvalCasesBoth:
    @parametrize(data=KEYPOINTS_DATA)
    def case_eval_keypoints(
        self, eval_keypoints_np: COCOevalnp, eval_keypoints_pt: COCOevalpt, data: tuple[int, int, Any]
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_keypoints_np, eval_keypoints_pt, img_id, cat_id, result)


@pytest.mark.benchmark(group="computeOks", warmup=True)
@parametrize_with_cases("coco_eval_np, img_id, cat_id, result", cases=COCOEvalCasesNp)
def test_computeOks_np(benchmark, coco_eval_np: COCOevalnp, img_id: int, cat_id: int, result):  # noqa: N802
    ious = coco_eval_np.computeOks(img_id, cat_id)
    # ious = benchmark(coco_eval_np.computeOks, img_id, cat_id)
    result = np.array(result)
    assert ious.shape == result.shape
    assert np.allclose(ious, result)


@pytest.mark.benchmark(group="computeOks", warmup=True)
@parametrize_with_cases("coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesPt)
def test_computeOks_pt(benchmark, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result):  # noqa: N802
    ious = coco_eval_pt.computeOks(img_id, cat_id)
    # ious = benchmark(coco_eval_pt.computeOks, img_id, cat_id)
    assert len(ious) == len(result)
    assert torch.allclose(ious, torch.Tensor(result))


@parametrize_with_cases("coco_eval_np, coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesBoth)
def test_ccomputeOks(coco_eval_np: COCOevalnp, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result):  # noqa: N802
    ious_np = coco_eval_np.computeOks(img_id, cat_id)
    ious_pt = coco_eval_pt.computeOks(img_id, cat_id)
    assert np.allclose(ious_np, ious_pt)
    assert torch.allclose(ious_pt, torch.Tensor(result))
