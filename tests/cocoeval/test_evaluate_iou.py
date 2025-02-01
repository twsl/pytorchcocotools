from typing import Any, TypeAlias, cast

import numpy as np
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize, parametrize_with_cases
import torch
from torch import Tensor

from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811

BBOX_DATA = [
    [
        [0.5],
        None,
        None,
        [0.5],
        None,
        None,
        [0.5],
        None,
        None,
        [0.5],
        None,
        None,
        [0.8],
        [0.7],
        None,
        [0.8],
        [0.7],
        None,
        [0.8],
        [0.7],
        None,
        [0.8],
        [0.7],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
]
SEGM_DATA = [
    [
        [0.5],
        None,
        None,
        [0.5],
        None,
        None,
        [0.5],
        None,
        None,
        [0.5],
        None,
        None,
        [0.8],
        [0.7],
        None,
        [0.8],
        [0.7],
        None,
        [0.8],
        [0.7],
        None,
        [0.8],
        [0.7],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
]


class COCOEvalCasesNp:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(self, eval_bbox_np: COCOevalnp, data: tuple[Any]) -> tuple[COCOevalnp, Any]:
        result = data
        return (eval_bbox_np, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(self, eval_segm_np: COCOevalnp, data: tuple[Any]) -> tuple[COCOevalnp, Any]:
        result = data
        return (eval_segm_np, result)


class COCOEvalCasesPt:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(self, eval_bbox_pt: COCOevalpt, data: tuple[Any]) -> tuple[COCOevalpt, Any]:
        result = data
        return (eval_bbox_pt, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(self, eval_segm_pt: COCOevalpt, data: tuple[Any]) -> tuple[COCOevalpt, Any]:
        result = data
        return (eval_segm_pt, result)


class COCOEvalCasesBoth:
    @parametrize(data=BBOX_DATA)
    def case_eval_bbox(
        self, eval_bbox_np: COCOevalnp, eval_bbox_pt: COCOevalpt, data: tuple[Any]
    ) -> tuple[COCOevalnp, COCOevalpt, Any]:
        result = data
        return (eval_bbox_np, eval_bbox_pt, result)

    @parametrize(data=SEGM_DATA)
    def case_eval_segm(
        self, eval_segm_np: COCOevalnp, eval_segm_pt: COCOevalpt, data: tuple[Any]
    ) -> tuple[COCOevalnp, COCOevalpt, Any]:
        result = data
        return (eval_segm_np, eval_segm_pt, result)


@pytest.mark.benchmark(group="evaluate", warmup=True)
@parametrize_with_cases("coco_eval_np, result", cases=COCOEvalCasesNp)
def test_evaluate_np(benchmark: BenchmarkFixture, coco_eval_np: COCOevalnp, result) -> None:  # noqa: N802
    # coco_eval_np.evaluate()
    benchmark(coco_eval_np.evaluate)
    assert len(coco_eval_np.evalImgs) == len(result)
    for i, (img, res) in enumerate(zip(coco_eval_np.evalImgs, result, strict=False)):
        if (img is None) ^ (res is None):
            pytest.fail(f"img[{i}] is None xor res[{i}] is None")
        if img is None and res is None:
            continue
        assert np.allclose(img["dtScores"], res)


@pytest.mark.benchmark(group="evaluate", warmup=True)
@parametrize_with_cases("coco_eval_pt, result", cases=COCOEvalCasesPt)
def test_evaluate_pt(benchmark: BenchmarkFixture, coco_eval_pt: COCOevalpt, result) -> None:  # noqa: N802
    # coco_eval_pt.evaluate()
    benchmark(coco_eval_pt.evaluate)
    assert len(coco_eval_pt.eval_imgs) == len(result)
    for i, (img, res) in enumerate(zip(coco_eval_pt.eval_imgs, result, strict=False)):
        if (img is None) ^ (res is None):
            pytest.fail(f"img[{i}] is None xor res[{i}] is None")
        if img is None and res is None:
            continue
        assert torch.allclose(img.dtScores, torch.tensor(res))  # pyright: ignore[reportOptionalMemberAccess]


@parametrize_with_cases("coco_eval_np, coco_eval_pt, result", cases=COCOEvalCasesBoth)
def test_evaluate(coco_eval_np: COCOevalnp, coco_eval_pt: COCOevalpt, result) -> None:  # noqa: N802
    coco_eval_np.evaluate()
    coco_eval_pt.evaluate()
    assert len(coco_eval_np.evalImgs) == len(coco_eval_pt.eval_imgs)
    assert len(coco_eval_np.evalImgs) == len(result)
    # assert np.allclose(ious_np, np.array(result))
    # assert torch.allclose(ious_pt, torch.tensor(result, dtype=torch.float32))
    for i, (img_pt, img_np, res) in enumerate(zip(coco_eval_pt.eval_imgs, coco_eval_np.evalImgs, result, strict=False)):
        if (img_pt is None) ^ (img_np is None):
            pytest.fail(f"img_pt[{i}] is None xor img_np[{i}] is None")
        if img_pt is None and img_np is None and res is None:
            continue
        assert torch.allclose(img_pt.dtScores, torch.tensor(res))  # pyright: ignore[reportOptionalMemberAccess]
        assert np.allclose(img_np["dtScores"], res)
