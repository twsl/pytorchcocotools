"""Unit tests for MeanAveragePrecision – compute() edge cases.

Covers empty inputs, missing predictions/ground-truths, no predictions,
and perfect predictions – each paired with a torchmetrics parity check
and benchmark variants.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
import torch
from torch import IntTensor, Tensor

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision


class TestComputeEdgeCases:
    # ---------------------------------------------------------------------------
    # Empty predictions
    # ---------------------------------------------------------------------------

    def test_empty_preds_does_not_raise(
        self,
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        m.update(empty_preds, target)
        result = m.compute()
        reference = tm_compute(deepcopy(empty_preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_empty_preds", warmup=True)
    def test_empty_preds_tm(self, benchmark: BenchmarkFixture, tm_compute: Callable[..., dict[str, Tensor]]) -> None:
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        benchmark(tm_compute, empty_preds, target, iou_type="bbox")

    @pytest.mark.benchmark(group="compute_empty_preds", warmup=True)
    def test_empty_preds_pt(self, benchmark: BenchmarkFixture, pt_compute: Callable[..., dict[str, Tensor]]) -> None:
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        benchmark(pt_compute, empty_preds, target, iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Empty ground truths
    # ---------------------------------------------------------------------------

    def test_empty_ground_truths_does_not_raise(
        self,
        bbox_preds: list[dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        m.update(bbox_preds, empty_target)
        result = m.compute()
        reference = tm_compute(deepcopy(bbox_preds), deepcopy(empty_target), iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_empty_gt", warmup=True)
    def test_empty_ground_truths_tm(
        self, benchmark: BenchmarkFixture, bbox_preds: list[dict[str, Tensor]], tm_compute: _ComputeFn
    ) -> None:
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        benchmark(tm_compute, bbox_preds, empty_target, iou_type="bbox")

    @pytest.mark.benchmark(group="compute_empty_gt", warmup=True)
    def test_empty_ground_truths_pt(
        self, benchmark: BenchmarkFixture, bbox_preds: list[dict[str, Tensor]], pt_compute: _ComputeFn
    ) -> None:
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        benchmark(pt_compute, bbox_preds, empty_target, iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Missing predictions
    # ---------------------------------------------------------------------------

    def test_missing_pred_map_less_than_one(
        self, pt_compute: _ComputeFn, tm_compute: _ComputeFn, assert_map_close: _AssertFn
    ) -> None:
        """When no preds given for any GT box, map should be 0 (not -1)."""
        preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[10.0, 20.0, 50.0, 60.0]]), "labels": IntTensor([0])}]
        result = pt_compute(preds, target, iou_type="bbox")
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)
        assert result["map"].item() < 1.0

    @pytest.mark.benchmark(group="compute_missing_preds", warmup=True)
    def test_missing_pred_tm(self, benchmark: BenchmarkFixture, tm_compute: Callable[..., dict[str, Tensor]]) -> None:
        preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[10.0, 20.0, 50.0, 60.0]]), "labels": IntTensor([0])}]
        benchmark(tm_compute, preds, target, iou_type="bbox")

    @pytest.mark.benchmark(group="compute_missing_preds", warmup=True)
    def test_missing_pred_pt(self, benchmark: BenchmarkFixture, pt_compute: Callable[..., dict[str, Tensor]]) -> None:
        preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[10.0, 20.0, 50.0, 60.0]]), "labels": IntTensor([0])}]
        benchmark(pt_compute, preds, target, iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Missing ground truths
    # ---------------------------------------------------------------------------

    def test_missing_gt_map_less_than_one(
        self,
        bbox_preds: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """When no GT boxes given for any pred, map should be 0 (not -1)."""
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        result = pt_compute(bbox_preds, empty_target, iou_type="bbox")
        reference = tm_compute(deepcopy(bbox_preds), deepcopy(empty_target), iou_type="bbox")
        assert_map_close(result, reference)
        assert result["map"].item() < 1.0

    @pytest.mark.benchmark(group="compute_missing_gt", warmup=True)
    def test_missing_gt_tm(
        self, benchmark: BenchmarkFixture, bbox_preds: list[dict[str, Tensor]], tm_compute: _ComputeFn
    ) -> None:
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        benchmark(tm_compute, bbox_preds, empty_target, iou_type="bbox")

    @pytest.mark.benchmark(group="compute_missing_gt", warmup=True)
    def test_missing_gt_pt(
        self, benchmark: BenchmarkFixture, bbox_preds: list[dict[str, Tensor]], pt_compute: _ComputeFn
    ) -> None:
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        benchmark(pt_compute, bbox_preds, empty_target, iou_type="bbox")

    # ---------------------------------------------------------------------------
    # No predictions
    # ---------------------------------------------------------------------------

    def test_no_predictions_returns_negative_one(
        self,
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """When there are no predictions, mAP and mAR should be -1."""
        empty_preds = [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int32)}
        ]
        result = pt_compute(empty_preds, bbox_target, iou_type="bbox")
        reference = tm_compute(empty_preds, bbox_target, iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_no_preds", warmup=True)
    def test_no_predictions_tm(
        self, benchmark: BenchmarkFixture, bbox_target: list[dict[str, Tensor]], tm_compute: _ComputeFn
    ) -> None:
        empty_preds = [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int32)}
        ]
        benchmark(tm_compute, empty_preds, bbox_target, iou_type="bbox")

    @pytest.mark.benchmark(group="compute_no_preds", warmup=True)
    def test_no_predictions_pt(
        self, benchmark: BenchmarkFixture, bbox_target: list[dict[str, Tensor]], pt_compute: _ComputeFn
    ) -> None:
        empty_preds = [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int32)}
        ]
        benchmark(pt_compute, empty_preds, bbox_target, iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Perfect predictions
    # ---------------------------------------------------------------------------

    def test_perfect_predictions_map_is_one(
        self, pt_compute: _ComputeFn, tm_compute: _ComputeFn, assert_map_close: _AssertFn
    ) -> None:
        """Exact predictions should yield mAP ≈ 1.0."""
        boxes = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        preds = [{"boxes": boxes.clone(), "scores": torch.tensor([1.0]), "labels": torch.tensor([0])}]
        target = [{"boxes": boxes.clone(), "labels": torch.tensor([0])}]
        result = pt_compute(preds, target, iou_type="bbox")
        reference = tm_compute(preds, target, iou_type="bbox")
        assert_map_close(result, reference)
        torch.testing.assert_close(result["map"].float(), torch.tensor(1.0), atol=1e-4, rtol=0.0)

    @pytest.mark.benchmark(group="compute_perfect", warmup=True)
    def test_perfect_predictions_tm(
        self, benchmark: BenchmarkFixture, tm_compute: Callable[..., dict[str, Tensor]]
    ) -> None:
        boxes = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        preds = [{"boxes": boxes.clone(), "scores": torch.tensor([1.0]), "labels": torch.tensor([0])}]
        target = [{"boxes": boxes.clone(), "labels": torch.tensor([0])}]
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    @pytest.mark.benchmark(group="compute_perfect", warmup=True)
    def test_perfect_predictions_pt(
        self, benchmark: BenchmarkFixture, pt_compute: Callable[..., dict[str, Tensor]]
    ) -> None:
        boxes = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        preds = [{"boxes": boxes.clone(), "scores": torch.tensor([1.0]), "labels": torch.tensor([0])}]
        target = [{"boxes": boxes.clone(), "labels": torch.tensor([0])}]
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")
