"""Stress tests for MeanAveragePrecision – compute() with large box counts.

Covers compute correctness and benchmarks at scale (25–50 boxes per image,
8 images, 10 classes).  Update stress tests live in test_stress_update.py.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
import torch
from torch import Tensor

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision


class TestStressCompute:
    def test_compute_incremental_vs_batch_large_pt(
        self,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """Incremental vs batch update must produce identical compute() results."""
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=30)

        m_batch = MeanAveragePrecision(iou_type="bbox")
        m_batch.update(preds, target)
        result_batch = m_batch.compute()

        m_inc = MeanAveragePrecision(iou_type="bbox")
        for p, t in zip(preds, target):
            m_inc.update([p], [t])
        result_inc = m_inc.compute()

        assert_map_close(result_batch, result_inc)

    # ---------------------------------------------------------------------------
    # Compute on 25 / 50 boxes per image
    # ---------------------------------------------------------------------------

    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_compute_25_50_boxes(
        self,
        n_boxes: int,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """compute() output on large batches must match torchmetrics reference."""
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=n_boxes)
        result = pt_compute(preds, target, iou_type="bbox")
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="stress_compute_boxes", warmup=True)
    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_compute_25_50_boxes_tm(
        self,
        benchmark: BenchmarkFixture,
        n_boxes: int,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=n_boxes)
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    @pytest.mark.benchmark(group="stress_compute_boxes", warmup=True)
    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_compute_25_50_boxes_pt(
        self,
        benchmark: BenchmarkFixture,
        n_boxes: int,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=n_boxes)
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Many images
    # ---------------------------------------------------------------------------

    def test_compute_many_images(
        self,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """compute() on 8 images x 30 boxes across 10 classes matches torchmetrics."""
        preds, target = make_stress_batch(n_images=8, n_boxes_per_image=30, n_classes=10)
        result = pt_compute(preds, target, iou_type="bbox")
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="stress_compute_many_images", warmup=True)
    def test_compute_many_images_tm(
        self,
        benchmark: BenchmarkFixture,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds, target = make_stress_batch(n_images=8, n_boxes_per_image=30, n_classes=10)
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    @pytest.mark.benchmark(group="stress_compute_many_images", warmup=True)
    def test_compute_many_images_pt(
        self,
        benchmark: BenchmarkFixture,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds, target = make_stress_batch(n_images=8, n_boxes_per_image=30, n_classes=10)
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Class metrics (large)
    # ---------------------------------------------------------------------------

    def test_compute_class_metrics_large(
        self,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """class_metrics=True with many classes and 40 boxes per image."""
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=40, n_classes=5)
        result = pt_compute(preds, target, iou_type="bbox", class_metrics=True)
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox", class_metrics=True)
        assert_map_close(result, reference)
        assert "map_per_class" in result
        torch.testing.assert_close(
            result["map_per_class"].float(),
            reference["map_per_class"].float(),
            atol=1e-4,
            rtol=0.0,
        )

    @pytest.mark.benchmark(group="stress_compute_class_metrics", warmup=True)
    def test_compute_class_metrics_large_tm(
        self,
        benchmark: BenchmarkFixture,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=40, n_classes=5)
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox", class_metrics=True)

    @pytest.mark.benchmark(group="stress_compute_class_metrics", warmup=True)
    def test_compute_class_metrics_large_pt(
        self,
        benchmark: BenchmarkFixture,
        make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=40, n_classes=5)
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox", class_metrics=True)
