from typing import Any, cast

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize, parametrize_with_cases
import torch
from torch import Tensor

from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811

KEYPOINTS_DATA = [
    (1, 4, [[0.85715781]]),
]


# class KeypointsCases:
#     # @case(id="start_area")
#     def case_start_area(self) -> tuple[int, int, Any]:
#         return (1, 1, [[1.0]])


class COCOEvalCasesNp:
    @parametrize(data=KEYPOINTS_DATA)
    # @parametrize_with_cases("data", cases=KeypointsCases)
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
def test_computeOks_np(benchmark: BenchmarkFixture, coco_eval_np: COCOevalnp, img_id: int, cat_id: int, result) -> None:  # noqa: N802
    ious = cast(np.ndarray, benchmark(coco_eval_np.computeOks, img_id, cat_id))
    result = np.array(result)
    assert ious.shape == result.shape
    assert np.allclose(ious, result)


@pytest.mark.benchmark(group="computeOks", warmup=True)
@parametrize_with_cases("coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesPt)
def test_computeOks_pt(benchmark: BenchmarkFixture, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result) -> None:  # noqa: N802
    ious = cast(Tensor, benchmark(coco_eval_pt.computeOks, img_id, cat_id))
    assert len(ious) == len(result)
    assert torch.allclose(ious, torch.tensor(result))


@parametrize_with_cases("coco_eval_np, coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesBoth)
def test_ccomputeOks(coco_eval_np: COCOevalnp, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result) -> None:  # noqa: N802
    ious_np = coco_eval_np.computeOks(img_id, cat_id)
    ious_pt = coco_eval_pt.computeOks(img_id, cat_id)
    assert np.allclose(ious_np, np.array(result))
    assert torch.allclose(ious_pt, torch.tensor(result))
