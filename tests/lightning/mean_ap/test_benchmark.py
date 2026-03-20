"""Benchmarks and parity tests for MeanAveragePrecision vs torchmetrics original.

Benchmark functions ending with ``_tm`` measure the upstream
``torchmetrics.detection.MeanAveragePrecision``; those ending with ``_pt``
measure our own ``pytorchcocotools.lightning.metrics.mean_ap.MeanAveragePrecision``.
The ``pytest-park`` grouping in ``conftest.py`` clusters the two together for
side-by-side comparison in the benchmark report.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
import torch
from torch import Tensor


class TestMeanAveragePrecisionBenchmark:
    """Benchmarks and parity tests for MeanAveragePrecision vs torchmetrics."""

    # ---------------------------------------------------------------------------
    # Parity – assert our implementation matches torchmetrics
    # ---------------------------------------------------------------------------

    def test_realistic_inputs_match(self, inputs: dict[str, list[list[dict[str, Tensor]]]], pt_compute: Callable[..., dict[str, Tensor]], tm_compute: Callable[..., dict[str, Tensor]], assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None]) -> None:
        preds = inputs["preds"][0] + inputs["preds"][1]
        target = inputs["target"][0] + inputs["target"][1]
        result = pt_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)

    def test_inputs2_empty_target_match(self, inputs2: dict[str, list[list[dict[str, Tensor]]]], pt_compute: Callable[..., dict[str, Tensor]], tm_compute: Callable[..., dict[str, Tensor]], assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None]) -> None:
        preds = inputs2["preds"][0] + inputs2["preds"][1]
        target = inputs2["target"][0] + inputs2["target"][1]
        result = pt_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)

    def test_inputs3_empty_preds_match(self, inputs3: dict[str, list[list[dict[str, Tensor]]]], pt_compute: Callable[..., dict[str, Tensor]], tm_compute: Callable[..., dict[str, Tensor]], assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None]) -> None:
        preds = inputs3["preds"][0] + inputs3["preds"][1]
        target = inputs3["target"][0] + inputs3["target"][1]
        result = pt_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.parametrize("n_boxes", [10, 25, 50])
    def test_stress_batch_match(self, n_boxes: int, make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]], pt_compute: Callable[..., dict[str, Tensor]], tm_compute: Callable[..., dict[str, Tensor]], assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None]) -> None:
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=n_boxes)
        result = pt_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)

    def test_class_metrics_match(self, make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]], pt_compute: Callable[..., dict[str, Tensor]], tm_compute: Callable[..., dict[str, Tensor]], assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None]) -> None:
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=25, n_classes=5)
        result = pt_compute(deepcopy(preds), deepcopy(target), iou_type="bbox", class_metrics=True)
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox", class_metrics=True)
        assert_map_close(result, reference)
        torch.testing.assert_close(
            result["map_per_class"].float(),
            reference["map_per_class"].float(),
            atol=1e-4,
            rtol=0.0,
        )

    # ---------------------------------------------------------------------------
    # Benchmarks – realistic inputs (small)
    # ---------------------------------------------------------------------------

    @pytest.mark.benchmark(group="mean_ap_realistic", warmup=True)
    def test_mean_ap_realistic_tm(self, benchmark: BenchmarkFixture, inputs: dict[str, list[list[dict[str, Tensor]]]], tm_compute: Callable[..., dict[str, Tensor]]) -> None:
        """Torchmetrics: update + compute on realistic multi-image inputs."""
        preds = inputs["preds"][0] + inputs["preds"][1]
        target = inputs["target"][0] + inputs["target"][1]
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    @pytest.mark.benchmark(group="mean_ap_realistic", warmup=True)
    def test_mean_ap_realistic_pt(self, benchmark: BenchmarkFixture, inputs: dict[str, list[list[dict[str, Tensor]]]], pt_compute: Callable[..., dict[str, Tensor]]) -> None:
        """Own implementation: update + compute on realistic multi-image inputs."""
        preds = inputs["preds"][0] + inputs["preds"][1]
        target = inputs["target"][0] + inputs["target"][1]
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Benchmarks – stress batches (parametrized by n_images, n_boxes)
    # ---------------------------------------------------------------------------

    @pytest.mark.benchmark(group="mean_ap_stress", warmup=True)
    @pytest.mark.parametrize("n_images,n_boxes", [(4, 10), (4, 25), (4, 50), (8, 25)])
    def test_mean_ap_stress_tm(self, benchmark: BenchmarkFixture, n_images: int, n_boxes: int, make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]], tm_compute: Callable[..., dict[str, Tensor]]) -> None:
        """Torchmetrics: update + compute on stress-test batch."""
        preds, target = make_stress_batch(n_images=n_images, n_boxes_per_image=n_boxes)
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    @pytest.mark.benchmark(group="mean_ap_stress", warmup=True)
    @pytest.mark.parametrize("n_images,n_boxes", [(4, 10), (4, 25), (4, 50), (8, 25)])
    def test_mean_ap_stress_pt(self, benchmark: BenchmarkFixture, n_images: int, n_boxes: int, make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]], pt_compute: Callable[..., dict[str, Tensor]]) -> None:
        """Own implementation: update + compute on stress-test batch."""
        preds, target = make_stress_batch(n_images=n_images, n_boxes_per_image=n_boxes)
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Benchmarks – class metrics enabled
    # ---------------------------------------------------------------------------

    @pytest.mark.benchmark(group="mean_ap_class_metrics", warmup=True)
    def test_mean_ap_class_metrics_tm(self, benchmark: BenchmarkFixture, make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]], tm_compute: Callable[..., dict[str, Tensor]]) -> None:
        """Torchmetrics: update + compute with class_metrics=True."""
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=25, n_classes=5)
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox", class_metrics=True)

    @pytest.mark.benchmark(group="mean_ap_class_metrics", warmup=True)
    def test_mean_ap_class_metrics_pt(self, benchmark: BenchmarkFixture, make_stress_batch: Callable[..., tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]], pt_compute: Callable[..., dict[str, Tensor]]) -> None:
        """Own implementation: update + compute with class_metrics=True."""
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=25, n_classes=5)
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox", class_metrics=True)
