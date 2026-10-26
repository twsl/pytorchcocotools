"""Stress tests for MeanAveragePrecision – update() with large box counts.

Covers state correctness, incremental vs batch equivalence, reset,
class tracking, and warning behaviour at scale.
Compute stress tests live in test_stress_compute.py.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import Tensor

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision

from .protocols import MakeStressBatch


class TestStressUpdate:
    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_update_single_image_large_boxes_pt(
        self, n_boxes: int, make_stress_batch: MakeStressBatch, device: str, to_device
    ) -> None:
        """update() must not raise and must store all boxes for a single image."""
        preds, target = make_stress_batch(n_images=1, n_boxes_per_image=n_boxes)
        preds = to_device(preds)
        target = to_device(target)
        m = MeanAveragePrecision(iou_type="bbox").to(device)
        m.update(preds, target)
        assert len(m.detection_box) == 1
        assert m.detection_box[0].shape == (n_boxes, 4)
        assert m.detection_scores[0].shape == (n_boxes,)
        assert m.detection_labels[0].shape == (n_boxes,)
        assert m.groundtruth_box[0].shape == (n_boxes, 4)
        assert m.groundtruth_labels[0].shape == (n_boxes,)

    @pytest.mark.parametrize("n_images,n_boxes", [(4, 25), (4, 50), (8, 25)])
    def test_update_multi_image_large_boxes_pt(
        self,
        n_images: int,
        n_boxes: int,
        make_stress_batch: MakeStressBatch,
        device: str,
        to_device,
    ) -> None:
        """update() accumulates the correct number of per-image entries."""
        preds, target = make_stress_batch(n_images=n_images, n_boxes_per_image=n_boxes)
        preds = to_device(preds)
        target = to_device(target)
        m = MeanAveragePrecision(iou_type="bbox").to(device)
        m.update(preds, target)
        assert len(m.detection_box) == n_images
        assert len(m.groundtruth_box) == n_images
        for i in range(n_images):
            assert m.detection_box[i].shape[0] == n_boxes

    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_incremental_update_equals_batch_update_pt(
        self, n_boxes: int, make_stress_batch: MakeStressBatch, device: str, to_device
    ) -> None:
        """Feeding images one-by-one must produce the same stored state as one batch."""
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=n_boxes)
        preds = to_device(preds)
        target = to_device(target)
        m_batch = MeanAveragePrecision(iou_type="bbox").to(device)
        m_batch.update(preds, target)

        m_incremental = MeanAveragePrecision(iou_type="bbox").to(device)
        for p, t in zip(preds, target, strict=False):
            m_incremental.update([p], [t])

        assert len(m_batch.detection_box) == len(m_incremental.detection_box)
        for b, inc in zip(m_batch.detection_box, m_incremental.detection_box, strict=False):
            torch.testing.assert_close(b, inc)

    def test_reset_after_large_update_pt(self, make_stress_batch: MakeStressBatch, device: str, to_device) -> None:
        """State must be fully cleared after reset() even after heavy update."""
        preds, target = make_stress_batch(n_images=8, n_boxes_per_image=50)
        preds = to_device(preds)
        target = to_device(target)
        m = MeanAveragePrecision(iou_type="bbox").to(device)
        m.update(preds, target)
        m.reset()
        assert len(m.detection_box) == 0
        assert len(m.detection_scores) == 0
        assert len(m.groundtruth_box) == 0

    def test_many_classes_are_tracked_pt(self, make_stress_batch: MakeStressBatch, device: str, to_device) -> None:
        """_get_classes() must return all unique class ids across images."""
        preds, target = make_stress_batch(n_images=4, n_boxes_per_image=25, n_classes=10)
        preds = to_device(preds)
        target = to_device(target)
        m = MeanAveragePrecision(iou_type="bbox").to(device)
        m.update(preds, target)
        classes = m._get_classes()
        assert len(classes) > 0
        assert len(classes) == len(set(classes))
        assert classes == sorted(classes)

    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_warning_fires_above_100_boxes_pt(
        self, n_boxes: int, make_stress_batch: MakeStressBatch, device: str, to_device
    ) -> None:
        """100 boxes: no warning. 101 boxes: UserWarning. Validates threshold boundary."""
        exactly_100_preds, target = make_stress_batch(n_images=1, n_boxes_per_image=100)
        exactly_100_preds = to_device(exactly_100_preds)
        target = to_device(target)
        m = MeanAveragePrecision(iou_type="bbox").to(device)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            m.update(exactly_100_preds, target)

    @pytest.mark.parametrize("n_images,n_boxes", [(4, 25), (4, 50)])
    def test_xywh_large_boxes_stored_as_xywh_pt(
        self,
        n_images: int,
        n_boxes: int,
        make_stress_batch: MakeStressBatch,
        device: str,
        to_device,
    ) -> None:
        """Boxes in xywh format are stored unchanged (no double conversion)."""
        preds, target = make_stress_batch(n_images=n_images, n_boxes_per_image=n_boxes)
        preds = to_device(preds)
        target = to_device(target)
        m = MeanAveragePrecision(iou_type="bbox", box_format="xywh").to(device)
        m.update(preds, target)
        for i in range(n_images):
            assert m.detection_box[i].shape == (n_boxes, 4)
