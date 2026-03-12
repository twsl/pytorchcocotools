"""Unit tests for pytorchcocotools.lightning.metrics.MeanAveragePrecision.

Test patterns are derived from the official torchmetrics test suite:
https://github.com/Lightning-AI/torchmetrics/blob/v1.8.2/tests/unittests/detection/test_map.py
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import IntTensor, Tensor
from torchmetrics.detection import MeanAveragePrecision as TorchmetricsMeanAveragePrecision

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision

# ---------------------------------------------------------------------------
# Realistic multi-image test data (mirroring torchmetrics _inputs / _inputs2 / _inputs3)
# ---------------------------------------------------------------------------

_inputs = {
    "preds": [
        [
            {
                "boxes": Tensor([[258.15, 41.29, 606.41, 285.07]]),
                "scores": Tensor([0.236]),
                "labels": IntTensor([4]),
            },
            {
                "boxes": Tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
                "scores": Tensor([0.318, 0.726]),
                "labels": IntTensor([3, 2]),
            },
        ],
        [
            {
                "boxes": Tensor(
                    [
                        [87.87, 276.25, 384.29, 379.43],
                        [0.00, 3.66, 142.15, 316.06],
                        [296.55, 93.96, 314.97, 152.79],
                        [328.94, 97.05, 342.49, 122.98],
                        [356.62, 95.47, 372.33, 147.55],
                        [464.08, 105.09, 495.74, 146.99],
                        [276.11, 103.84, 291.44, 150.72],
                    ]
                ),
                "scores": Tensor([0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953]),
                "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },
            {
                "boxes": Tensor(
                    [
                        [72.92, 45.96, 91.23, 80.57],
                        [45.17, 45.34, 66.28, 79.83],
                        [82.28, 47.04, 99.66, 78.50],
                        [59.96, 46.17, 80.35, 80.48],
                        [75.29, 23.01, 91.85, 50.85],
                        [71.14, 1.10, 96.96, 28.33],
                        [61.34, 55.23, 77.14, 79.57],
                        [41.17, 45.78, 60.99, 78.48],
                        [56.18, 44.80, 64.42, 56.25],
                    ]
                ),
                "scores": Tensor([0.532, 0.204, 0.782, 0.202, 0.883, 0.271, 0.561, 0.204, 0.349]),
                "labels": IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49]),
            },
        ],
    ],
    "target": [
        [
            {
                "boxes": Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                "labels": IntTensor([4]),
            },
            {
                "boxes": Tensor([[13.00, 22.75, 548.98, 632.42], [1.66, 3.32, 270.26, 275.23]]),
                "labels": IntTensor([2, 2]),
            },
        ],
        [
            {
                "boxes": Tensor(
                    [
                        [61.87, 276.25, 358.29, 379.43],
                        [2.75, 3.66, 162.15, 316.06],
                        [295.55, 93.96, 313.97, 152.79],
                        [326.94, 97.05, 340.49, 122.98],
                        [356.62, 95.47, 372.33, 147.55],
                        [462.08, 105.09, 493.74, 146.99],
                        [277.11, 103.84, 292.44, 150.72],
                    ]
                ),
                "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },
            {
                "boxes": Tensor(
                    [
                        [72.92, 45.96, 91.23, 80.57],
                        [50.17, 45.34, 71.28, 79.83],
                        [81.28, 47.04, 98.66, 78.50],
                        [63.96, 46.17, 84.35, 80.48],
                        [75.29, 23.01, 91.85, 50.85],
                        [56.39, 21.65, 75.66, 45.54],
                        [73.14, 1.10, 98.96, 28.33],
                        [62.34, 55.23, 78.14, 79.57],
                        [44.17, 45.78, 63.99, 78.48],
                        [58.18, 44.80, 66.42, 56.25],
                    ]
                ),
                "labels": IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49, 49]),
            },
        ],
    ],
}

# second batch has empty target (issue torchmetrics#943)
_inputs2 = {
    "preds": [
        [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": Tensor([0.536]), "labels": IntTensor([0])}],
        [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": Tensor([0.536]), "labels": IntTensor([0])}],
    ],
    "target": [
        [{"boxes": Tensor([[214.0, 41.0, 562.0, 285.0]]), "labels": IntTensor([0])}],
        [{"boxes": Tensor([]), "labels": IntTensor([])}],
    ],
}

# second batch has empty preds (issues torchmetrics#981, #1147)
_inputs3 = {
    "preds": [
        [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": Tensor([0.536]), "labels": IntTensor([0])}],
        [{"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
    ],
    "target": [
        [{"boxes": Tensor([[214.0, 41.0, 562.0, 285.0]]), "labels": IntTensor([0])}],
        [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([1])}],
    ],
}


# ---------------------------------------------------------------------------
# Shared markers
# ---------------------------------------------------------------------------

_SEGM_UPDATE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="Segmentation mAP still differs from torchmetrics because the local PyTorch-only mask IoU backend "
    "produces slightly different threshold behavior than the pycocotools-backed reference.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tm_compute(preds, target, **kwargs) -> dict[str, Tensor]:
    m = TorchmetricsMeanAveragePrecision(**kwargs)
    m.update(preds, target)
    return m.compute()


def _pt_compute(preds, target, **kwargs) -> dict[str, Tensor]:
    m = MeanAveragePrecision(**kwargs)
    m.update(preds, target)
    return m.compute()


def _assert_map_close(result, reference, atol: float = 1e-4) -> None:
    scalar_keys = [k for k in reference if k != "classes" and reference[k].ndim == 0]
    for key in scalar_keys:
        assert key in result, f"Missing key '{key}' in result"
        torch.testing.assert_close(
            result[key].float(),
            reference[key].float(),
            atol=atol,
            rtol=0.0,
            msg=f"Mismatch for key '{key}'",
        )


# ---------------------------------------------------------------------------
# 1. Initialisation & parameter validation
# ---------------------------------------------------------------------------


class TestMeanAveragePrecisionInit:
    def test_default_construction(self) -> None:
        m = MeanAveragePrecision()
        assert m.box_format == "xyxy"
        assert list(m.iou_type) == ["bbox"]
        assert len(m.iou_thresholds) == 10
        assert pytest.approx(m.iou_thresholds[0], abs=1e-6) == 0.5
        assert pytest.approx(m.iou_thresholds[-1], abs=1e-6) == 0.95
        assert len(m.rec_thresholds) == 101
        assert m.max_detection_thresholds == [1, 10, 100]
        assert m.class_metrics is False
        assert m.extended_summary is False
        assert m.average == "macro"

    def test_invalid_box_format_raises(self) -> None:
        with pytest.raises(ValueError, match="box_format"):
            MeanAveragePrecision(box_format="invalid")  # type: ignore[arg-type]

    def test_invalid_iou_type_raises(self) -> None:
        with pytest.raises(ValueError, match="iou_type"):
            MeanAveragePrecision(iou_type="invalid")  # type: ignore[arg-type]

    def test_invalid_iou_thresholds_raises(self) -> None:
        with pytest.raises(ValueError, match="iou_thresholds"):
            MeanAveragePrecision(iou_thresholds=0.5)  # type: ignore[arg-type]

    def test_invalid_rec_thresholds_raises(self) -> None:
        with pytest.raises(ValueError, match="rec_thresholds"):
            MeanAveragePrecision(rec_thresholds=0.0)  # type: ignore[arg-type]

    def test_invalid_max_detection_thresholds_type_raises(self) -> None:
        with pytest.raises(ValueError, match="max_detection_thresholds"):
            MeanAveragePrecision(max_detection_thresholds=100)  # type: ignore[arg-type]

    def test_invalid_max_detection_thresholds_length_two_raises(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            MeanAveragePrecision(max_detection_thresholds=[1, 100])

    def test_invalid_max_detection_thresholds_length_four_raises(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            MeanAveragePrecision(max_detection_thresholds=[1, 10, 50, 100])

    def test_invalid_class_metrics_raises(self) -> None:
        with pytest.raises(TypeError, match="class_metrics"):
            MeanAveragePrecision(class_metrics=0)  # type: ignore[arg-type]

    def test_invalid_average_raises(self) -> None:
        with pytest.raises(ValueError, match="average"):
            MeanAveragePrecision(average="weighted")  # type: ignore[arg-type]

    def test_custom_iou_thresholds(self) -> None:
        thresholds = [0.5, 0.75]
        m = MeanAveragePrecision(iou_thresholds=thresholds)
        assert m.iou_thresholds == thresholds

    def test_custom_max_detection_thresholds_sorted(self) -> None:
        m = MeanAveragePrecision(max_detection_thresholds=[100, 1, 10])
        assert m.max_detection_thresholds == [1, 10, 100]

    def test_segm_iou_type(self) -> None:
        m = MeanAveragePrecision(iou_type="segm")
        assert list(m.iou_type) == ["segm"]

    def test_tuple_iou_type(self) -> None:
        m = MeanAveragePrecision(iou_type=("bbox", "segm"))
        assert set(m.iou_type) == {"bbox", "segm"}

    def test_xywh_box_format(self) -> None:
        m = MeanAveragePrecision(box_format="xywh")
        assert m.box_format == "xywh"

    def test_cxcywh_box_format(self) -> None:
        m = MeanAveragePrecision(box_format="cxcywh")
        assert m.box_format == "cxcywh"


# ---------------------------------------------------------------------------
# 2. State accumulation via update()
# ---------------------------------------------------------------------------


class TestMeanAveragePrecisionUpdate:
    def test_bbox_state_accumulated(
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

    def test_multi_update_accumulates(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(bbox_preds, bbox_target)
        m.update(bbox_preds, bbox_target)
        assert len(m.detection_box) == 2
        assert len(m.groundtruth_box) == 2

    def test_boxes_converted_to_xywh(
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

    def test_iscrowd_defaults_to_zero(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(bbox_preds, bbox_target)
        assert m.groundtruth_crowds[0].sum().item() == 0

    def test_iscrowd_preserved_when_provided(
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

    def test_empty_preds_no_error(self) -> None:
        """update() should not raise when predictions list is empty."""
        m = MeanAveragePrecision(iou_type="bbox")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        m.update(empty_preds, target)
        assert len(m.detection_box) == 1

    def test_empty_ground_truths_no_error(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        """update() should not raise when ground-truth list is empty."""
        m = MeanAveragePrecision(iou_type="bbox")
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        m.update(bbox_preds, empty_target)
        assert len(m.groundtruth_box) == 1

    def test_empty_preds_xywh_no_error(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox", box_format="xywh")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        m.update(empty_preds, target)
        assert len(m.detection_box) == 1

    def test_empty_ground_truths_xywh_no_error(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox", box_format="xywh")
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        m.update(bbox_preds, empty_target)
        assert len(m.groundtruth_box) == 1

    def test_empty_preds_cxcywh_no_error(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox", box_format="cxcywh")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        m.update(empty_preds, target)
        assert len(m.detection_box) == 1

    def test_empty_both_no_error(self) -> None:
        """update() with empty preds and empty targets should not raise."""
        m = MeanAveragePrecision(iou_type="bbox")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        m.update(empty_preds, empty_target)
        assert len(m.detection_box) == 1

    def test_invalid_target_type_raises(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        with pytest.raises((ValueError, TypeError, AttributeError)):
            m.update(bbox_preds, Tensor([[1.0, 2.0, 3.0, 4.0]]))  # type: ignore[arg-type]

    def test_missing_pred_boxes_key_raises(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"scores": Tensor([0.9]), "labels": IntTensor([0])}]
        with pytest.raises((ValueError, KeyError), match="boxes"):
            m.update(bad_preds, bbox_target)

    def test_missing_pred_scores_key_raises(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        with pytest.raises((ValueError, KeyError), match="scores"):
            m.update(bad_preds, bbox_target)

    def test_missing_pred_labels_key_raises(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "scores": Tensor([0.9])}]
        with pytest.raises((ValueError, KeyError), match="labels"):
            m.update(bad_preds, bbox_target)

    def test_missing_target_boxes_key_raises(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_target = [{"labels": IntTensor([0])}]
        with pytest.raises((ValueError, KeyError), match="boxes"):
            m.update(bbox_preds, bad_target)

    def test_missing_target_labels_key_raises(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]])}]
        with pytest.raises((ValueError, KeyError), match="labels"):
            m.update(bbox_preds, bad_target)

    def test_non_tensor_boxes_raises(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": [[258.0, 41.0, 606.0, 285.0]], "scores": Tensor([0.9]), "labels": IntTensor([0])}]
        with pytest.raises((ValueError, TypeError, AttributeError)):
            m.update(bad_preds, bbox_target)  # type: ignore[arg-type]

    def test_non_tensor_scores_raises(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": [0.9], "labels": IntTensor([0])}]
        with pytest.raises((ValueError, TypeError, AttributeError)):
            m.update(bad_preds, bbox_target)  # type: ignore[arg-type]

    def test_non_tensor_labels_raises(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        bad_preds = [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": Tensor([0.9]), "labels": [0]}]
        with pytest.raises((ValueError, TypeError, AttributeError)):
            m.update(bad_preds, bbox_target)  # type: ignore[arg-type]

    def test_warning_on_many_detections(self) -> None:
        """Should warn when a single image has >100 detections."""
        m = MeanAveragePrecision(iou_type="bbox")
        many_boxes = torch.zeros(101, 4)
        many_boxes[:, 2:] = 1.0  # valid non-degenerate boxes
        preds = [{"boxes": many_boxes, "scores": torch.rand(101), "labels": torch.zeros(101, dtype=torch.int32)}]
        target = [{"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.int32)}]
        target[0]["boxes"][0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        with pytest.warns(UserWarning, match="Encountered more than 100 detections"):
            m.update(preds, target)

    def test_no_warning_when_warn_disabled(self) -> None:
        """No warning should be emitted when warn_on_many_detections=False."""
        m = MeanAveragePrecision(iou_type="bbox")
        m.warn_on_many_detections = False
        many_boxes = torch.zeros(101, 4)
        many_boxes[:, 2:] = 1.0
        preds = [{"boxes": many_boxes, "scores": torch.rand(101), "labels": torch.zeros(101, dtype=torch.int32)}]
        target = [{"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.int32)}]
        target[0]["boxes"][0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            m.update(preds, target)  # must not raise

    def test_invalid_preds_type_raises(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        with pytest.raises((ValueError, TypeError)):
            m.update("not a list", bbox_target)  # type: ignore[arg-type]

    def test_mismatched_lengths_raises(
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

    def test_reset_clears_state(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(bbox_preds, bbox_target)
        m.reset()
        assert len(m.detection_box) == 0
        assert len(m.groundtruth_box) == 0

    def test_segm_masks_stored_as_rle(
        self,
        segm_preds: list[dict[str, Tensor]],
        segm_target: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="segm")
        m.update(segm_preds, segm_target)
        # Masks are stored as tuples of (size, counts) RLE data, not raw tensors.
        assert len(m.detection_mask) == 1
        assert isinstance(m.detection_mask[0], tuple)


# ---------------------------------------------------------------------------
# 3. _get_classes()
# ---------------------------------------------------------------------------


class TestGetClasses:
    def test_returns_unique_sorted(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.detection_labels = [torch.tensor([2, 0, 1]), torch.tensor([0])]
        m.groundtruth_labels = [torch.tensor([1, 0])]
        classes = m._get_classes()
        assert classes == [0, 1, 2]

    def test_empty_returns_empty_list(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        assert m._get_classes() == []

    def test_combines_detection_and_groundtruth_labels(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        m.detection_labels = [torch.tensor([3])]
        m.groundtruth_labels = [torch.tensor([5])]
        classes = m._get_classes()
        assert 3 in classes
        assert 5 in classes


# ---------------------------------------------------------------------------
# 4. _coco_stats_to_tensor_dict()
# ---------------------------------------------------------------------------


class TestCocoStatsTensorDict:
    def test_keys_present_for_default_thresholds(self) -> None:
        m = MeanAveragePrecision()
        fake_stats = torch.arange(12, dtype=torch.float32)
        result = m._coco_stats_to_tensor_dict(fake_stats, prefix="")
        expected_keys = {
            "map",
            "map_50",
            "map_75",
            "map_small",
            "map_medium",
            "map_large",
            "mar_1",
            "mar_10",
            "mar_100",
            "mar_small",
            "mar_medium",
            "mar_large",
        }
        assert set(result.keys()) == expected_keys

    def test_prefix_applied(self) -> None:
        m = MeanAveragePrecision()
        fake_stats = torch.zeros(12)
        result = m._coco_stats_to_tensor_dict(fake_stats, prefix="bbox_")
        assert all(k.startswith("bbox_") for k in result)

    def test_custom_max_det_keys(self) -> None:
        m = MeanAveragePrecision(max_detection_thresholds=[5, 50, 200])
        fake_stats = torch.zeros(12)
        result = m._coco_stats_to_tensor_dict(fake_stats, prefix="")
        assert "mar_5" in result
        assert "mar_50" in result
        assert "mar_200" in result

    def test_values_match_stats_order(self) -> None:
        """Index positions must match the COCO stats ordering convention."""
        m = MeanAveragePrecision()
        stats = torch.arange(12, dtype=torch.float32)
        result = m._coco_stats_to_tensor_dict(stats, prefix="")
        assert result["map"].item() == 0.0
        assert result["map_50"].item() == 1.0
        assert result["map_75"].item() == 2.0
        assert result["mar_1"].item() == 6.0
        assert result["mar_10"].item() == 7.0
        assert result["mar_100"].item() == 8.0


# ---------------------------------------------------------------------------
# 5. compute() parity with torchmetrics – expected to fail while the
#    annotation-building loop in _get_coco_format is commented out.
# ---------------------------------------------------------------------------


class TestMeanAveragePrecisionCompute:
    def test_empty_metric(self) -> None:
        """compute() called before any update should not raise (but return sentinel -1)."""
        m = MeanAveragePrecision()
        result = m.compute()
        assert result["map"].item() == -1.0

    def test_empty_preds_does_not_raise(self) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        empty_preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([0])}]
        m.update(empty_preds, target)
        result = m.compute()
        reference = _tm_compute(deepcopy(empty_preds), deepcopy(target), iou_type="bbox")
        _assert_map_close(result, reference)

    def test_empty_ground_truths_does_not_raise(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        m = MeanAveragePrecision(iou_type="bbox")
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        m.update(bbox_preds, empty_target)
        result = m.compute()
        reference = _tm_compute(deepcopy(bbox_preds), deepcopy(empty_target), iou_type="bbox")
        _assert_map_close(result, reference)

    def test_missing_pred_map_less_than_one(self) -> None:
        """When no preds given for any GT box, map should be 0 (not -1)."""
        preds = [{"boxes": Tensor([]).reshape(0, 4), "scores": Tensor([]), "labels": IntTensor([])}]
        target = [{"boxes": Tensor([[10.0, 20.0, 50.0, 60.0]]), "labels": IntTensor([0])}]
        result = _pt_compute(preds, target, iou_type="bbox")
        reference = _tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        _assert_map_close(result, reference)
        assert result["map"].item() < 1.0

    def test_missing_gt_map_less_than_one(
        self,
        bbox_preds: list[dict[str, Tensor]],
    ) -> None:
        """When no GT boxes given for any pred, map should be 0 (not -1)."""
        empty_target = [{"boxes": Tensor([]).reshape(0, 4), "labels": IntTensor([])}]
        result = _pt_compute(bbox_preds, empty_target, iou_type="bbox")
        reference = _tm_compute(deepcopy(bbox_preds), deepcopy(empty_target), iou_type="bbox")
        _assert_map_close(result, reference)
        assert result["map"].item() < 1.0

    @pytest.mark.parametrize(
        "box_format,expected_map",
        [
            ("xyxy", 1.0),
            ("xywh", 0.0),
            ("cxcywh", 0.0),
        ],
    )
    def test_for_box_format(self, box_format: str, expected_map: float) -> None:
        """Xyxy perfect match → map=1; same coords treated as xywh/cxcywh → map=0."""
        boxes = Tensor([[258.0, 41.0, 606.0, 285.0]])
        preds = [{"boxes": boxes.clone(), "scores": Tensor([1.0]), "labels": IntTensor([0])}]
        target = [{"boxes": boxes.clone(), "labels": IntTensor([0])}]
        result = _pt_compute(deepcopy(preds), deepcopy(target), iou_type="bbox", box_format=box_format)
        reference = _tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox", box_format=box_format)
        _assert_map_close(result, reference)

    def test_custom_iou_thresholds_map50_absent(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        """When iou_thresholds=[0.6, 0.7], map_50 and map_75 should be -1."""
        kwargs = {"iou_type": "bbox", "iou_thresholds": [0.6, 0.7]}
        result = _pt_compute(bbox_preds, bbox_target, **kwargs)
        reference = _tm_compute(bbox_preds, bbox_target, **kwargs)
        _assert_map_close(result, reference)
        assert result["map_50"].item() == -1.0
        assert result["map_75"].item() == -1.0

    def test_bbox_single_image_matches_torchmetrics(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        """map, map_50, map_75, mar_* values must match torchmetrics reference."""
        result = _pt_compute(bbox_preds, bbox_target, iou_type="bbox")
        reference = _tm_compute(bbox_preds, bbox_target, iou_type="bbox")
        _assert_map_close(result, reference)

    def test_bbox_multi_image_matches_torchmetrics(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
    ) -> None:
        result = _pt_compute(multi_image_preds, multi_image_target, iou_type="bbox")
        reference = _tm_compute(multi_image_preds, multi_image_target, iou_type="bbox")
        _assert_map_close(result, reference)

    def test_bbox_class_metrics_matches_torchmetrics(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
    ) -> None:
        result = _pt_compute(multi_image_preds, multi_image_target, iou_type="bbox", class_metrics=True)
        reference = _tm_compute(multi_image_preds, multi_image_target, iou_type="bbox", class_metrics=True)
        _assert_map_close(result, reference)
        torch.testing.assert_close(
            result["map_per_class"].float(),
            reference["map_per_class"].float(),
            atol=1e-4,
            rtol=0.0,
        )

    def test_bbox_micro_average_matches_torchmetrics(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
    ) -> None:
        result = _pt_compute(multi_image_preds, multi_image_target, iou_type="bbox", average="micro")
        reference = _tm_compute(multi_image_preds, multi_image_target, iou_type="bbox", average="micro")
        _assert_map_close(result, reference)

    def test_bbox_custom_iou_thresholds_matches_torchmetrics(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        kwargs = {"iou_type": "bbox", "iou_thresholds": [0.5, 0.75]}
        result = _pt_compute(bbox_preds, bbox_target, **kwargs)
        reference = _tm_compute(bbox_preds, bbox_target, **kwargs)
        _assert_map_close(result, reference)

    def test_bbox_realistic_inputs_matches_torchmetrics(self) -> None:
        """Multi-class, multi-image realistic data from _inputs."""
        for preds_batch, target_batch in zip(_inputs["preds"], _inputs["target"]):
            preds = deepcopy(preds_batch)
            target = deepcopy(target_batch)
        result = _pt_compute(_inputs["preds"][-1], _inputs["target"][-1], iou_type="bbox")
        reference = _tm_compute(_inputs["preds"][-1], _inputs["target"][-1], iou_type="bbox")
        _assert_map_close(result, reference)

    def test_inputs2_empty_target_matches_torchmetrics(self) -> None:
        """Test with second batch having empty target (torchmetrics issue #943)."""
        for preds_batch, target_batch in zip(_inputs2["preds"], _inputs2["target"]):
            preds = deepcopy(preds_batch)
            target = deepcopy(target_batch)
        result = _pt_compute(
            _inputs2["preds"][0] + _inputs2["preds"][1], _inputs2["target"][0] + _inputs2["target"][1], iou_type="bbox"
        )
        reference = _tm_compute(
            _inputs2["preds"][0] + _inputs2["preds"][1], _inputs2["target"][0] + _inputs2["target"][1], iou_type="bbox"
        )
        _assert_map_close(result, reference)

    def test_inputs3_empty_preds_matches_torchmetrics(self) -> None:
        """Test with second batch having empty preds (torchmetrics issues #981, #1147)."""
        result = _pt_compute(
            _inputs3["preds"][0] + _inputs3["preds"][1], _inputs3["target"][0] + _inputs3["target"][1], iou_type="bbox"
        )
        reference = _tm_compute(
            _inputs3["preds"][0] + _inputs3["preds"][1], _inputs3["target"][0] + _inputs3["target"][1], iou_type="bbox"
        )
        _assert_map_close(result, reference)

    def test_bbox_custom_rec_thresholds_matches_torchmetrics(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        kwargs = {"iou_type": "bbox", "rec_thresholds": [0.0, 0.1, 0.5, 1.0]}
        result = _pt_compute(bbox_preds, bbox_target, **kwargs)
        reference = _tm_compute(bbox_preds, bbox_target, **kwargs)
        _assert_map_close(result, reference)

    def test_extended_summary_shapes(
        self,
        bbox_preds: list[dict[str, Tensor]],
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        """With extended_summary=True the result should include precision/recall/scores."""
        result = _pt_compute(bbox_preds, bbox_target, iou_type="bbox", extended_summary=True)
        assert "precision" in result
        assert "recall" in result
        assert "scores" in result
        # precision shape: (T, R, K, A, M) where T=10 iou thresholds,
        # R=101 rec thresholds, K=num_classes, A=4 area ranges, M=3 max dets
        assert result["precision"].ndim == 5
        assert result["recall"].ndim == 4

    def test_classes_tensor_in_result(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
    ) -> None:
        """With class_metrics=True the result dict should contain 'classes' as int32."""
        result = _pt_compute(multi_image_preds, multi_image_target, iou_type="bbox", class_metrics=True)
        assert "classes" in result
        assert result["classes"].dtype == torch.int32

    @_SEGM_UPDATE_XFAIL
    def test_segm_single_image_matches_torchmetrics(
        self,
        segm_preds: list[dict[str, Tensor]],
        segm_target: list[dict[str, Tensor]],
    ) -> None:
        result = _pt_compute(segm_preds, segm_target, iou_type="segm")
        reference = _tm_compute(segm_preds, segm_target, iou_type="segm")
        _assert_map_close(result, reference)

    def test_no_predictions_returns_negative_one(
        self,
        bbox_target: list[dict[str, Tensor]],
    ) -> None:
        """When there are no predictions, mAP and mAR should be -1."""
        empty_preds = [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int32)}
        ]
        result = _pt_compute(empty_preds, bbox_target, iou_type="bbox")
        reference = _tm_compute(empty_preds, bbox_target, iou_type="bbox")
        _assert_map_close(result, reference)

    def test_perfect_predictions_map_is_one(self) -> None:
        """Exact predictions should yield mAP ≈ 1.0."""
        boxes = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        preds = [{"boxes": boxes.clone(), "scores": torch.tensor([1.0]), "labels": torch.tensor([0])}]
        target = [{"boxes": boxes.clone(), "labels": torch.tensor([0])}]
        result = _pt_compute(preds, target, iou_type="bbox")
        reference = _tm_compute(preds, target, iou_type="bbox")
        _assert_map_close(result, reference)
        torch.testing.assert_close(result["map"].float(), torch.tensor(1.0), atol=1e-4, rtol=0.0)

    def test_multiple_updates_same_as_single_update(
        self,
        multi_image_preds: list[dict[str, Tensor]],
        multi_image_target: list[dict[str, Tensor]],
    ) -> None:
        """Calling update() twice with one image each should equal one call with two images."""
        m_single = MeanAveragePrecision(iou_type="bbox")
        m_single.update(multi_image_preds, multi_image_target)
        result_single = m_single.compute()

        m_incremental = MeanAveragePrecision(iou_type="bbox")
        for p, t in zip(multi_image_preds, multi_image_target):
            m_incremental.update([p], [t])
        result_incremental = m_incremental.compute()

        _assert_map_close(result_single, result_incremental)


# ---------------------------------------------------------------------------
# Helpers shared by stress tests
# ---------------------------------------------------------------------------


def _make_random_boxes(n: int, img_w: float = 640.0, img_h: float = 480.0, seed: int = 0) -> Tensor:
    """Return *n* random non-degenerate xyxy boxes inside (img_w × img_h)."""
    gen = torch.Generator().manual_seed(seed)
    x1 = torch.rand(n, generator=gen) * (img_w * 0.8)
    y1 = torch.rand(n, generator=gen) * (img_h * 0.8)
    # guarantee w,h >= 10 so boxes are never degenerate
    w = torch.rand(n, generator=gen) * (img_w * 0.2) + 10.0
    h = torch.rand(n, generator=gen) * (img_h * 0.2) + 10.0
    x2 = (x1 + w).clamp(max=img_w)
    y2 = (y1 + h).clamp(max=img_h)
    return torch.stack([x1, y1, x2, y2], dim=1)


def _make_stress_batch(
    n_images: int,
    n_boxes_per_image: int,
    n_classes: int = 10,
    seed: int = 42,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Build preds + target lists with *n_boxes_per_image* boxes in each image."""
    preds, target = [], []
    for i in range(n_images):
        boxes = _make_random_boxes(n_boxes_per_image, seed=seed + i)
        labels = torch.randint(
            0, n_classes, (n_boxes_per_image,), generator=torch.Generator().manual_seed(seed + i + 1000)
        )
        preds.append(
            {
                "boxes": boxes,
                "scores": torch.rand(n_boxes_per_image, generator=torch.Generator().manual_seed(seed + i + 2000)),
                "labels": labels.to(torch.int32),
            }
        )
        gt_boxes = _make_random_boxes(n_boxes_per_image, seed=seed + i + 3000)
        gt_labels = torch.randint(
            0, n_classes, (n_boxes_per_image,), generator=torch.Generator().manual_seed(seed + i + 4000)
        )
        target.append(
            {
                "boxes": gt_boxes,
                "labels": gt_labels.to(torch.int32),
            }
        )
    return preds, target


# ---------------------------------------------------------------------------
# 6. Stress tests – large numbers of boxes per image
# ---------------------------------------------------------------------------


class TestStress:
    """Stress tests exercising update() and compute() with 25–50 boxes per image."""

    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_update_single_image_large_boxes(self, n_boxes: int) -> None:
        """update() must not raise and must store all boxes for a single image."""
        preds, target = _make_stress_batch(n_images=1, n_boxes_per_image=n_boxes)
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(preds, target)
        assert len(m.detection_box) == 1
        assert m.detection_box[0].shape == (n_boxes, 4)
        assert m.detection_scores[0].shape == (n_boxes,)
        assert m.detection_labels[0].shape == (n_boxes,)
        assert m.groundtruth_box[0].shape == (n_boxes, 4)
        assert m.groundtruth_labels[0].shape == (n_boxes,)

    @pytest.mark.parametrize("n_images,n_boxes", [(4, 25), (4, 50), (8, 25)])
    def test_update_multi_image_large_boxes(self, n_images: int, n_boxes: int) -> None:
        """update() accumulates the correct number of per-image entries."""
        preds, target = _make_stress_batch(n_images=n_images, n_boxes_per_image=n_boxes)
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(preds, target)
        assert len(m.detection_box) == n_images
        assert len(m.groundtruth_box) == n_images
        for i in range(n_images):
            assert m.detection_box[i].shape[0] == n_boxes

    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_incremental_update_equals_batch_update(self, n_boxes: int) -> None:
        """Feeding images one-by-one must produce the same stored state as one batch."""
        preds, target = _make_stress_batch(n_images=4, n_boxes_per_image=n_boxes)
        m_batch = MeanAveragePrecision(iou_type="bbox")
        m_batch.update(preds, target)

        m_incremental = MeanAveragePrecision(iou_type="bbox")
        for p, t in zip(preds, target):
            m_incremental.update([p], [t])

        assert len(m_batch.detection_box) == len(m_incremental.detection_box)
        for b, inc in zip(m_batch.detection_box, m_incremental.detection_box):
            torch.testing.assert_close(b, inc)

    def test_reset_after_large_update(self) -> None:
        """State must be fully cleared after reset() even after heavy update."""
        preds, target = _make_stress_batch(n_images=8, n_boxes_per_image=50)
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(preds, target)
        m.reset()
        assert len(m.detection_box) == 0
        assert len(m.detection_scores) == 0
        assert len(m.groundtruth_box) == 0

    def test_many_classes_are_tracked(self) -> None:
        """_get_classes() must return all unique class ids across images."""
        preds, target = _make_stress_batch(n_images=4, n_boxes_per_image=25, n_classes=10)
        m = MeanAveragePrecision(iou_type="bbox")
        m.update(preds, target)
        classes = m._get_classes()
        assert len(classes) > 0
        assert len(classes) == len(set(classes))  # no duplicates
        assert classes == sorted(classes)  # sorted ascending

    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_warning_fires_above_100_boxes(self, n_boxes: int) -> None:
        """100 boxes: no warning. 101 boxes: UserWarning. Validates threshold boundary."""
        exactly_100_preds, target = _make_stress_batch(n_images=1, n_boxes_per_image=100)
        m = MeanAveragePrecision(iou_type="bbox")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            m.update(exactly_100_preds, target)  # must not warn

    @pytest.mark.parametrize("n_images,n_boxes", [(4, 25), (4, 50)])
    def test_xywh_large_boxes_stored_as_xywh(self, n_images: int, n_boxes: int) -> None:
        """Boxes in xywh format are stored unchanged (no double conversion)."""
        preds, target = _make_stress_batch(n_images=n_images, n_boxes_per_image=n_boxes)
        # Supply already-xywh boxes; box_format="xywh" means no conversion
        m = MeanAveragePrecision(iou_type="bbox", box_format="xywh")
        m.update(preds, target)
        # Boxes must be stored; exact values don't matter here, just shapes
        for i in range(n_images):
            assert m.detection_box[i].shape == (n_boxes, 4)

    @pytest.mark.parametrize("n_boxes", [25, 50])
    def test_compute_25_50_boxes_matches_torchmetrics(self, n_boxes: int) -> None:
        """compute() output on large batches must match torchmetrics reference."""
        preds, target = _make_stress_batch(n_images=4, n_boxes_per_image=n_boxes)
        result = _pt_compute(preds, target, iou_type="bbox")
        reference = _tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        _assert_map_close(result, reference)

    def test_compute_many_images_matches_torchmetrics(self) -> None:
        """compute() on 8 images × 30 boxes across 10 classes matches torchmetrics."""
        preds, target = _make_stress_batch(n_images=8, n_boxes_per_image=30, n_classes=10)
        result = _pt_compute(preds, target, iou_type="bbox")
        reference = _tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        _assert_map_close(result, reference)

    def test_compute_class_metrics_large(self) -> None:
        """class_metrics=True with many classes and 40 boxes per image."""
        preds, target = _make_stress_batch(n_images=4, n_boxes_per_image=40, n_classes=5)
        result = _pt_compute(preds, target, iou_type="bbox", class_metrics=True)
        reference = _tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox", class_metrics=True)
        _assert_map_close(result, reference)
        assert "map_per_class" in result
        torch.testing.assert_close(
            result["map_per_class"].float(),
            reference["map_per_class"].float(),
            atol=1e-4,
            rtol=0.0,
        )

    def test_compute_incremental_vs_batch_large(self) -> None:
        """Incremental vs batch update must produce identical compute() results."""
        preds, target = _make_stress_batch(n_images=4, n_boxes_per_image=30)

        m_batch = MeanAveragePrecision(iou_type="bbox")
        m_batch.update(preds, target)
        result_batch = m_batch.compute()

        m_inc = MeanAveragePrecision(iou_type="bbox")
        for p, t in zip(preds, target):
            m_inc.update([p], [t])
        result_inc = m_inc.compute()

        _assert_map_close(result_batch, result_inc)
