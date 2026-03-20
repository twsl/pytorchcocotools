"""Unit tests for MeanAveragePrecision – pt-only internal compute() API.

These tests cover pytorchcocotools-specific behaviour with no torchmetrics equivalent.
Parity tests against torchmetrics live in test_compute_edge_cases.py,
test_compute_bbox.py, and test_compute_fixtures.py.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision


class TestComputeInternal:
    def test_empty_metric_pt(self) -> None:
        """compute() called before any update should not raise (but return sentinel -1)."""
        m = MeanAveragePrecision()
        result = m.compute()
        assert result["map"].item() == -1.0

    def test_extended_summary_shapes_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        """With extended_summary=True the result should include precision/recall/scores."""
        result = pt_compute(bbox_preds, bbox_target, iou_type="bbox", extended_summary=True)
        assert "precision" in result
        assert "recall" in result
        assert "scores" in result
        assert result["precision"].ndim == 5
        assert result["recall"].ndim == 4

    def test_classes_tensor_in_result_pt(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        """With class_metrics=True the result dict should contain 'classes' as int32."""
        result = pt_compute(multi_image_preds, multi_image_target, iou_type="bbox", class_metrics=True)
        assert "classes" in result
        assert result["classes"].dtype == torch.int32

    def test_multiple_updates_same_as_single_update_pt(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        """Calling update() twice with one image each should equal one call with two images."""
        m_single = MeanAveragePrecision(iou_type="bbox")
        m_single.update(multi_image_preds, multi_image_target)
        result_single = m_single.compute()

        m_incremental = MeanAveragePrecision(iou_type="bbox")
        for p, t in zip(multi_image_preds, multi_image_target):
            m_incremental.update([p], [t])
        result_incremental = m_incremental.compute()

        assert_map_close(result_single, result_incremental)
