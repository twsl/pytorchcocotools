"""Unit tests for MeanAveragePrecision – _coco_stats_to_tensor_dict()."""

from __future__ import annotations

import torch

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision


class TestCocoStatsTensorDict:
    def test_keys_present_for_default_thresholds_pt(self) -> None:
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

    def test_prefix_applied_pt(self) -> None:
        m = MeanAveragePrecision()
        fake_stats = torch.zeros(12)
        result = m._coco_stats_to_tensor_dict(fake_stats, prefix="bbox_")
        assert all(k.startswith("bbox_") for k in result)

    def test_custom_max_det_keys_pt(self) -> None:
        m = MeanAveragePrecision(max_detection_thresholds=[5, 50, 200])
        fake_stats = torch.zeros(12)
        result = m._coco_stats_to_tensor_dict(fake_stats, prefix="")
        assert "mar_5" in result
        assert "mar_50" in result
        assert "mar_200" in result

    def test_values_match_stats_order_pt(self) -> None:
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
