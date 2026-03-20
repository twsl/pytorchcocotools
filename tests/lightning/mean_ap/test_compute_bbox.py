"""Unit tests for MeanAveragePrecision – compute() bbox parity with torchmetrics.

Covers box formats, IoU/recall thresholds, single-image, multi-image,
class metrics, micro averaging, and realistic inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
import torch
from torch import IntTensor, Tensor


class TestComputeBbox:
    # ---------------------------------------------------------------------------
    # Box format
    # ---------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "box_format,expected_map",
        [("xyxy", 1.0), ("xywh", 0.0), ("cxcywh", 0.0)],
    )
    def test_for_box_format(
        self,
        box_format: str,
        expected_map: float,
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """Xyxy perfect match → map=1; same coords treated as xywh/cxcywh → map=0."""
        boxes = Tensor([[258.0, 41.0, 606.0, 285.0]])
        preds = [{"boxes": boxes.clone(), "scores": Tensor([1.0]), "labels": IntTensor([0])}]
        target = [{"boxes": boxes.clone(), "labels": IntTensor([0])}]
        result = pt_compute(deepcopy(preds), deepcopy(target), iou_type="bbox", box_format=box_format)
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox", box_format=box_format)
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_box_format", warmup=True)
    @pytest.mark.parametrize("box_format,expected_map", [("xyxy", 1.0), ("xywh", 0.0), ("cxcywh", 0.0)])
    def test_for_box_format_tm(
        self,
        benchmark: BenchmarkFixture,
        box_format: str,
        expected_map: float,
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        boxes = Tensor([[258.0, 41.0, 606.0, 285.0]])
        preds = [{"boxes": boxes.clone(), "scores": Tensor([1.0]), "labels": IntTensor([0])}]
        target = [{"boxes": boxes.clone(), "labels": IntTensor([0])}]
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox", box_format=box_format)

    @pytest.mark.benchmark(group="compute_box_format", warmup=True)
    @pytest.mark.parametrize("box_format,expected_map", [("xyxy", 1.0), ("xywh", 0.0), ("cxcywh", 0.0)])
    def test_for_box_format_pt(
        self,
        benchmark: BenchmarkFixture,
        box_format: str,
        expected_map: float,
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        boxes = Tensor([[258.0, 41.0, 606.0, 285.0]])
        preds = [{"boxes": boxes.clone(), "scores": Tensor([1.0]), "labels": IntTensor([0])}]
        target = [{"boxes": boxes.clone(), "labels": IntTensor([0])}]
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox", box_format=box_format)

    # ---------------------------------------------------------------------------
    # Custom IoU thresholds (map_50 absent)
    # ---------------------------------------------------------------------------

    def test_custom_iou_thresholds_map50_absent(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """When iou_thresholds=[0.6, 0.7], map_50 and map_75 should be -1."""
        kwargs = {"iou_type": "bbox", "iou_thresholds": [0.6, 0.7]}
        result = pt_compute(bbox_preds, bbox_target, **kwargs)
        reference = tm_compute(bbox_preds, bbox_target, **kwargs)
        assert_map_close(result, reference)
        assert result["map_50"].item() == -1.0
        assert result["map_75"].item() == -1.0

    @pytest.mark.benchmark(group="compute_custom_iou_absent", warmup=True)
    def test_custom_iou_thresholds_map50_absent_tm(
        self,
        benchmark: BenchmarkFixture,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(tm_compute, bbox_preds, bbox_target, iou_type="bbox", iou_thresholds=[0.6, 0.7])

    @pytest.mark.benchmark(group="compute_custom_iou_absent", warmup=True)
    def test_custom_iou_thresholds_map50_absent_pt(
        self,
        benchmark: BenchmarkFixture,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(pt_compute, bbox_preds, bbox_target, iou_type="bbox", iou_thresholds=[0.6, 0.7])

    # ---------------------------------------------------------------------------
    # Single image bbox
    # ---------------------------------------------------------------------------

    def test_bbox_single_image(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """map, map_50, map_75, mar_* values must match torchmetrics reference."""
        result = pt_compute(bbox_preds, bbox_target, iou_type="bbox")
        reference = tm_compute(bbox_preds, bbox_target, iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_bbox_single", warmup=True)
    def test_bbox_single_image_tm(
        self,
        benchmark: BenchmarkFixture,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(tm_compute, bbox_preds, bbox_target, iou_type="bbox")

    @pytest.mark.benchmark(group="compute_bbox_single", warmup=True)
    def test_bbox_single_image_pt(
        self,
        benchmark: BenchmarkFixture,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(pt_compute, bbox_preds, bbox_target, iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Multi-image bbox
    # ---------------------------------------------------------------------------

    def test_bbox_multi_image(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        result = pt_compute(multi_image_preds, multi_image_target, iou_type="bbox")
        reference = tm_compute(multi_image_preds, multi_image_target, iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_bbox_multi", warmup=True)
    def test_bbox_multi_image_tm(
        self,
        benchmark: BenchmarkFixture,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(tm_compute, multi_image_preds, multi_image_target, iou_type="bbox")

    @pytest.mark.benchmark(group="compute_bbox_multi", warmup=True)
    def test_bbox_multi_image_pt(
        self,
        benchmark: BenchmarkFixture,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(pt_compute, multi_image_preds, multi_image_target, iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Class metrics
    # ---------------------------------------------------------------------------

    def test_bbox_class_metrics(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        result = pt_compute(multi_image_preds, multi_image_target, iou_type="bbox", class_metrics=True)
        reference = tm_compute(multi_image_preds, multi_image_target, iou_type="bbox", class_metrics=True)
        assert_map_close(result, reference)
        torch.testing.assert_close(
            result["map_per_class"].float(),
            reference["map_per_class"].float(),
            atol=1e-4,
            rtol=0.0,
        )

    @pytest.mark.benchmark(group="compute_bbox_class_metrics", warmup=True)
    def test_bbox_class_metrics_tm(
        self,
        benchmark: BenchmarkFixture,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(tm_compute, multi_image_preds, multi_image_target, iou_type="bbox", class_metrics=True)

    @pytest.mark.benchmark(group="compute_bbox_class_metrics", warmup=True)
    def test_bbox_class_metrics_pt(
        self,
        benchmark: BenchmarkFixture,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(pt_compute, multi_image_preds, multi_image_target, iou_type="bbox", class_metrics=True)

    # ---------------------------------------------------------------------------
    # Micro average
    # ---------------------------------------------------------------------------

    def test_bbox_micro_average(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        result = pt_compute(multi_image_preds, multi_image_target, iou_type="bbox", average="micro")
        reference = tm_compute(multi_image_preds, multi_image_target, iou_type="bbox", average="micro")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_bbox_micro_average", warmup=True)
    def test_bbox_micro_average_tm(
        self,
        benchmark: BenchmarkFixture,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(tm_compute, multi_image_preds, multi_image_target, iou_type="bbox", average="micro")

    @pytest.mark.benchmark(group="compute_bbox_micro_average", warmup=True)
    def test_bbox_micro_average_pt(
        self,
        benchmark: BenchmarkFixture,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(pt_compute, multi_image_preds, multi_image_target, iou_type="bbox", average="micro")

    # ---------------------------------------------------------------------------
    # Custom IoU thresholds
    # ---------------------------------------------------------------------------

    def test_bbox_custom_iou_thresholds(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        kwargs = {"iou_type": "bbox", "iou_thresholds": [0.5, 0.75]}
        result = pt_compute(bbox_preds, bbox_target, **kwargs)
        reference = tm_compute(bbox_preds, bbox_target, **kwargs)
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_bbox_custom_iou", warmup=True)
    def test_bbox_custom_iou_thresholds_tm(
        self,
        benchmark: BenchmarkFixture,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(tm_compute, bbox_preds, bbox_target, iou_type="bbox", iou_thresholds=[0.5, 0.75])

    @pytest.mark.benchmark(group="compute_bbox_custom_iou", warmup=True)
    def test_bbox_custom_iou_thresholds_pt(
        self,
        benchmark: BenchmarkFixture,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(pt_compute, bbox_preds, bbox_target, iou_type="bbox", iou_thresholds=[0.5, 0.75])

    # ---------------------------------------------------------------------------
    # Custom recall thresholds
    # ---------------------------------------------------------------------------

    def test_bbox_custom_rec_thresholds(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        kwargs = {"iou_type": "bbox", "rec_thresholds": [0.0, 0.1, 0.5, 1.0]}
        result = pt_compute(bbox_preds, bbox_target, **kwargs)
        reference = tm_compute(bbox_preds, bbox_target, **kwargs)
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_bbox_custom_rec", warmup=True)
    def test_bbox_custom_rec_thresholds_tm(
        self,
        benchmark: BenchmarkFixture,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(tm_compute, bbox_preds, bbox_target, iou_type="bbox", rec_thresholds=[0.0, 0.1, 0.5, 1.0])

    @pytest.mark.benchmark(group="compute_bbox_custom_rec", warmup=True)
    def test_bbox_custom_rec_thresholds_pt(
        self,
        benchmark: BenchmarkFixture,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(pt_compute, bbox_preds, bbox_target, iou_type="bbox", rec_thresholds=[0.0, 0.1, 0.5, 1.0])

    # ---------------------------------------------------------------------------
    # Realistic inputs
    # ---------------------------------------------------------------------------

    def test_bbox_realistic_inputs(
        self,
        inputs: dict[str, list[list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """Multi-class, multi-image realistic data from inputs."""
        result = pt_compute(inputs["preds"][-1], inputs["target"][-1], iou_type="bbox")
        reference = tm_compute(inputs["preds"][-1], inputs["target"][-1], iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_bbox_realistic", warmup=True)
    def test_bbox_realistic_inputs_tm(
        self,
        benchmark: BenchmarkFixture,
        inputs: dict[str, list[list[dict[str, Tensor]]]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(tm_compute, inputs["preds"][-1], inputs["target"][-1], iou_type="bbox")

    @pytest.mark.benchmark(group="compute_bbox_realistic", warmup=True)
    def test_bbox_realistic_inputs_pt(
        self,
        benchmark: BenchmarkFixture,
        inputs: dict[str, list[list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(pt_compute, inputs["preds"][-1], inputs["target"][-1], iou_type="bbox")
