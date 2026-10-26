from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO as COCOnp  # noqa: N811
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
import torch

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811
from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811
from pytorchcocotools.internal.entities import IoUType

type E2EFiles = tuple[IoUType, Path, Path]


def _run_np(iou_type: IoUType, gt_path: Path, dt_path: Path) -> np.ndarray:
    coco = COCOnp(str(gt_path))
    coco_eval = COCOevalnp(coco, coco.loadRes(str(dt_path)), iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return np.asarray(coco_eval.stats)


def _run_pt(iou_type: IoUType, gt_path: Path, dt_path: Path, device: str) -> torch.Tensor:
    coco = COCOpt(str(gt_path), enable_logging=False, device=device)
    coco_eval = COCOevalpt(coco, coco.loadRes(str(dt_path)), iou_type, enable_logging=False, device=device)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def _run_hotcoco(hotcoco: Any, iou_type: IoUType, gt_path: Path, dt_path: Path) -> np.ndarray:
    coco = hotcoco.COCO(str(gt_path))
    coco_eval = hotcoco.COCOeval(coco, coco.load_res(str(dt_path)), iou_type)
    coco_eval.run()
    return np.asarray(coco_eval.stats)


@pytest.fixture(scope="session")
def e2e_reference(e2e_files: E2EFiles) -> np.ndarray:
    return _run_np(*e2e_files)


@pytest.mark.benchmark(group="e2e")
def test_e2e_np(benchmark: BenchmarkFixture, e2e_files: E2EFiles) -> None:
    benchmark.pedantic(_run_np, args=e2e_files, rounds=3, iterations=1, warmup_rounds=0)


@pytest.mark.benchmark(group="e2e")
def test_e2e_pt(benchmark: BenchmarkFixture, e2e_files: E2EFiles, e2e_reference: np.ndarray, device: str) -> None:
    stats = benchmark.pedantic(_run_pt, args=(*e2e_files, device), rounds=3, iterations=1, warmup_rounds=1)

    # float32 on MPS can flip matches whose IoU sits exactly on a threshold
    atol = 1e-6 if device == "cpu" else 2e-3
    np.testing.assert_allclose(stats.detach().cpu().double().numpy(), e2e_reference, atol=atol)


@pytest.mark.benchmark(group="e2e")
def test_e2e_hotcoco(benchmark: BenchmarkFixture, e2e_files: E2EFiles, e2e_reference: np.ndarray) -> None:
    hotcoco = pytest.importorskip("hotcoco", reason="hotcoco is an optional reference dependency")
    stats = benchmark.pedantic(_run_hotcoco, args=(hotcoco, *e2e_files), rounds=3, iterations=1, warmup_rounds=1)
    np.testing.assert_allclose(stats, e2e_reference, rtol=0, atol=1e-12)
