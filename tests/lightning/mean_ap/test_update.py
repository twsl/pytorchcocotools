"""Unit tests for MeanAveragePrecision – state accumulation via update().

Validation / error-raising tests live in test_update_validation.py.
"""

from __future__ import annotations

import torch
from torch import IntTensor, Tensor

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision


class TestMeanAveragePrecisionUpdate:
    def test_bbox_state_accumulated_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(bbox_preds, bbox_target)
        assert len(m.detection_box) == 1
        assert len(m.detection_labels) == 1
        assert len(m.detection_scores) == 1
        assert len(m.groundtruth_box) == 1
        assert len(m.groundtruth_labels) == 1

    def test_multi_update_accumulates_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(bbox_preds, bbox_target)
        m.update(bbox_preds, bbox_target)
        assert len(m.detection_box) == 2
        assert len(m.groundtruth_box) == 2

    def test_boxes_converted_to_xywh_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        """Boxes should be stored in xywh format regardless of box_format."""
        m = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
        m.update(bbox_preds, bbox_target)
        stored = m.detection_box[0]
        # xyxy [258, 41, 606, 285] → xywh [258, 41, 348, 244]
        expected = torch.tensor([[258.0, 41.0, 348.0, 244.0]])
        torch.testing.assert_close(stored, expected)

    def test_iscrowd_defaults_to_zero_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(bbox_preds, bbox_target)
        assert m.groundtruth_crowds[0].sum().item() == 0

    def test_iscrowd_preserved_when_provided_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        target = [
            {
                "boxes": torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": torch.tensor([0]),
                "iscrowd": torch.tensor([1]),
            }
        ]
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(bbox_preds, target)
        assert m.groundtruth_crowds[0][0].item() == 1

    def test_empty_preds_no_error_pt(self) -> None:
        """update() should not raise when predictions list is empty."""
        m = MeanAveragePrecision(iou_type="bbox")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        m.update(empty_preds, target)
        assert len(m.detection_box) == 1

    def test_empty_ground_truths_no_error_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        """update() should not raise when ground-truth list is empty."""
        m = MeanAveragePrecision(iou_type="bbox")
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        m.update(bbox_preds, empty_target)
        assert len(m.groundtruth_box) == 1

    def test_empty_preds_xywh_no_error_pt(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox", box_format="xywh")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        m.update(empty_preds, target)
        assert len(m.detection_box) == 1

    def test_empty_ground_truths_xywh_no_error_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox", box_format="xywh")
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        m.update(bbox_preds, empty_target)
        assert len(m.groundtruth_box) == 1

    def test_empty_preds_cxcywh_no_error_pt(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox", box_format="cxcywh")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        m.update(empty_preds, target)
        assert len(m.detection_box) == 1

    def test_empty_both_no_error_pt(self) -> None:
        """update() with empty preds and empty targets should not raise."""
        m = MeanAveragePrecision(iou_type="bbox")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        m.update(empty_preds, empty_target)
        assert len(m.detection_box) == 1

    def test_reset_clears_state_pt(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(bbox_preds, bbox_target)
        m.reset()
        assert len(m.detection_box) == 0
        assert len(m.groundtruth_box) == 0

    def test_segm_masks_stored_as_rle_pt(
        self,
        segm_preds: list[dict[str, Tensor]],
        segm_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="segm")
        m.update(segm_preds, segm_target)
        # Masks are stored as tuples of (size, counts) RLE data, not raw tensors.
        assert len(m.detection_mask) == 1
        assert isinstance(m.detection_mask[0], tuple)
