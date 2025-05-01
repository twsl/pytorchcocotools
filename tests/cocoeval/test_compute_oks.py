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


class KeypointsCases:
    # @case(id="test_1")
    def case_test_1(self) -> TEST_DATA:
        return (1, 4, [[0.85715781]])


class COCOEvalCasesNp:
    @parametrize_with_cases("data", cases=KeypointsCases)
    def case_eval_keypoints(self, eval_keypoints_np: COCOevalnp, data: TEST_DATA) -> tuple[COCOevalnp, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_keypoints_np, img_id, cat_id, result)


class COCOEvalCasesPt:
    @parametrize_with_cases("data", cases=KeypointsCases)
    def case_eval_keypoints(self, eval_keypoints_pt: COCOevalpt, data: TEST_DATA) -> tuple[COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_keypoints_pt, img_id, cat_id, result)


class COCOEvalCasesBoth:
    @parametrize_with_cases("data", cases=KeypointsCases)
    def case_eval_keypoints(
        self, eval_keypoints_np: COCOevalnp, eval_keypoints_pt: COCOevalpt, data: TEST_DATA
    ) -> tuple[COCOevalnp, COCOevalpt, int, int, Any]:
        img_id, cat_id, result = data
        return (eval_keypoints_np, eval_keypoints_pt, img_id, cat_id, result)


@pytest.mark.benchmark(group="computeOks", warmup=True)
@parametrize_with_cases("coco_eval_np, img_id, cat_id, result", cases=COCOEvalCasesNp)
def test_computeOks_np(  # noqa: N802
    benchmark: BenchmarkFixture, coco_eval_np: COCOevalnp, img_id: int, cat_id: int, result: Any
) -> None:
    ious = cast(np.ndarray, benchmark(coco_eval_np.computeOks, img_id, cat_id))
    result = np.array(result)
    assert ious.shape == result.shape
    assert np.allclose(ious, result)


@pytest.mark.benchmark(group="computeOks", warmup=True)
@parametrize_with_cases("coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesPt)
def test_computeOks_pt(  # noqa: N802
    benchmark: BenchmarkFixture, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result: Any
) -> None:
    ious = cast(Tensor, benchmark(coco_eval_pt.computeOks, img_id, cat_id))
    assert len(ious) == len(result)
    assert torch.allclose(ious, torch.tensor(result))


@parametrize_with_cases("coco_eval_np, coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCasesBoth)
def test_ccomputeOks(coco_eval_np: COCOevalnp, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result: Any) -> None:  # noqa: N802
    ious_np = coco_eval_np.computeOks(img_id, cat_id)
    ious_pt = coco_eval_pt.computeOks(img_id, cat_id)
    assert np.allclose(ious_np, np.array(result))
    assert torch.allclose(ious_pt, torch.tensor(result))
