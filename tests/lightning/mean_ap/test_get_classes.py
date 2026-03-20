"""Unit tests for MeanAveragePrecision – _get_classes()."""

from __future__ import annotations

import torch

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision


class TestGetClasses:
    def test_returns_unique_sorted_pt(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.detection_labels = [torch.tensor([2, 0, 1]), torch.tensor([0])]
        m.groundtruth_labels = [torch.tensor([1, 0])]
        classes = m._get_classes()
        assert classes == [0, 1, 2]

    def test_empty_returns_empty_list_pt(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        assert m._get_classes() == []

    def test_combines_detection_and_groundtruth_labels_pt(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.detection_labels = [torch.tensor([3])]
        m.groundtruth_labels = [torch.tensor([5])]
        classes = m._get_classes()
        assert 3 in classes
        assert 5 in classes
