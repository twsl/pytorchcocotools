"""Unit tests for MeanAveragePrecision – update() input validation."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import IntTensor, Tensor

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision


class TestMeanAveragePrecisionUpdateValidation:
    def test_invalid_target_type_raises_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        with pytest.raises((ValueError, TypeError, AttributeError)):
            m.update(bbox_preds, Tensor([[1.0, 2.0, 3.0, 4.0]]))  # type: ignore[arg-type]

    def test_missing_pred_boxes_key_raises_pt(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"scores": Tensor([0.9]), "labels": IntTensor([0])}]
        with pytest.raises((ValueError, KeyError), match="boxes"):
            m.update(bad_preds, bbox_target)

    def test_missing_pred_scores_key_raises_pt(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        with pytest.raises((ValueError, KeyError), match="scores"):
            m.update(bad_preds, bbox_target)

    def test_missing_pred_labels_key_raises_pt(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "scores": Tensor([0.9])}]
        with pytest.raises((ValueError, KeyError), match="labels"):
            m.update(bad_preds, bbox_target)

    def test_missing_target_boxes_key_raises_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_target = [{"labels": IntTensor([0])}]
        with pytest.raises((ValueError, KeyError), match="boxes"):
            m.update(bbox_preds, bad_target)

    def test_missing_target_labels_key_raises_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]])}]
        with pytest.raises((ValueError, KeyError), match="labels"):
            m.update(bbox_preds, bad_target)

    def test_non_tensor_boxes_raises_pt(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": [[258.0, 41.0, 606.0, 285.0]], "scores": Tensor([0.9]), "labels": IntTensor([0])}]
        with pytest.raises((ValueError, TypeError, AttributeError)):
            m.update(bad_preds, bbox_target)  # type: ignore[arg-type]

    def test_non_tensor_scores_raises_pt(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": [0.9], "labels": IntTensor([0])}]
        with pytest.raises((ValueError, TypeError, AttributeError)):
            m.update(bad_preds, bbox_target)  # type: ignore[arg-type]

    def test_non_tensor_labels_raises_pt(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": Tensor([0.9]), "labels": [0]}]
        with pytest.raises((ValueError, TypeError, AttributeError)):
            m.update(bad_preds, bbox_target)  # type: ignore[arg-type]

    def test_warning_on_many_detections_pt(self) -> None:
        """Should warn when a single image has >100 detections."""
        m = MeanAveragePrecision(iou_type="bbox")
        many_boxes = torch.zeros(101, 4)
        many_boxes[:, 2:] = 1.0  # valid non-degenerate boxes
        preds = [{"boxes": many_boxes, "scores": torch.rand(101), "labels": torch.zeros(101, dtype=torch.int32)}]
        target = [{"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.int32)}]
        target[0]["boxes"][0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        with pytest.warns(UserWarning, match="Encountered more than 100 detections"):
            m.update(preds, target)

    def test_no_warning_when_warn_disabled_pt(self) -> None:
        """No warning should be emitted when warn_on_many_detections=False."""
        m = MeanAveragePrecision(iou_type="bbox")
        m.warn_on_many_detections = False
        many_boxes = torch.zeros(101, 4)
        many_boxes[:, 2:] = 1.0
        preds = [{"boxes": many_boxes, "scores": torch.rand(101), "labels": torch.zeros(101, dtype=torch.int32)}]
        target = [{"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.int32)}]
        target[0]["boxes"][0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            m.update(preds, target)  # must not raise

    def test_invalid_preds_type_raises_pt(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        with pytest.raises((ValueError, TypeError)):
            m.update("not a list", bbox_target)  # type: ignore[arg-type]

    def test_mismatched_lengths_raises_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        target_two_images = [
            {"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.int32)},
            {"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.int32)},
        ]
        with pytest.raises(ValueError):
            m.update(bbox_preds, target_two_images)
