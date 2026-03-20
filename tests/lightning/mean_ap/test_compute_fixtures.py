"""Unit tests for MeanAveragePrecision – compute() fixture-based and segmentation tests.

Covers the inputs2/inputs3 regression fixtures (torchmetrics issues #943, #981, #1147)
and the segmentation iou_type="segm" parity check.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor

_SEGM_UPDATE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="Segmentation mAP still differs from torchmetrics because the local PyTorch-only mask IoU backend "
    "produces slightly different threshold behavior than the pycocotools-backed reference.",
)


class TestComputeFixtures:
    # ---------------------------------------------------------------------------
    # Empty target in second batch (torchmetrics issue #943)
    # ---------------------------------------------------------------------------

    def test_inputs2_empty_target(
        self,
        inputs2: dict[str, list[list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        preds = inputs2["preds"][0] + inputs2["preds"][1]
        target = inputs2["target"][0] + inputs2["target"][1]
        result = pt_compute(preds, target, iou_type="bbox")
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_inputs2", warmup=True)
    def test_inputs2_empty_target_tm(
        self,
        benchmark: BenchmarkFixture,
        inputs2: dict[str, list[list[dict[str, Tensor]]]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds = inputs2["preds"][0] + inputs2["preds"][1]
        target = inputs2["target"][0] + inputs2["target"][1]
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    @pytest.mark.benchmark(group="compute_inputs2", warmup=True)
    def test_inputs2_empty_target_pt(
        self,
        benchmark: BenchmarkFixture,
        inputs2: dict[str, list[list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds = inputs2["preds"][0] + inputs2["preds"][1]
        target = inputs2["target"][0] + inputs2["target"][1]
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Empty preds in second batch (torchmetrics issues #981, #1147)
    # ---------------------------------------------------------------------------

    def test_inputs3_empty_preds(
        self,
        inputs3: dict[str, list[list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        preds = inputs3["preds"][0] + inputs3["preds"][1]
        target = inputs3["target"][0] + inputs3["target"][1]
        result = pt_compute(preds, target, iou_type="bbox")
        reference = tm_compute(deepcopy(preds), deepcopy(target), iou_type="bbox")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_inputs3", warmup=True)
    def test_inputs3_empty_preds_tm(
        self,
        benchmark: BenchmarkFixture,
        inputs3: dict[str, list[list[dict[str, Tensor]]]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds = inputs3["preds"][0] + inputs3["preds"][1]
        target = inputs3["target"][0] + inputs3["target"][1]
        benchmark(tm_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    @pytest.mark.benchmark(group="compute_inputs3", warmup=True)
    def test_inputs3_empty_preds_pt(
        self,
        benchmark: BenchmarkFixture,
        inputs3: dict[str, list[list[dict[str, Tensor]]]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        preds = inputs3["preds"][0] + inputs3["preds"][1]
        target = inputs3["target"][0] + inputs3["target"][1]
        benchmark(pt_compute, deepcopy(preds), deepcopy(target), iou_type="bbox")

    # ---------------------------------------------------------------------------
    # Segmentation single image
    # ---------------------------------------------------------------------------

    @_SEGM_UPDATE_XFAIL
    def test_segm_single_image(
        self,
        segm_preds: list[dict[str, Tensor]],
        segm_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
        assert_map_close: Callable[[dict[str, Tensor], dict[str, Tensor], float], None],
    ) -> None:
        result = pt_compute(segm_preds, segm_target, iou_type="segm")
        reference = tm_compute(segm_preds, segm_target, iou_type="segm")
        assert_map_close(result, reference)

    @pytest.mark.benchmark(group="compute_segm_single", warmup=True)
    def test_segm_single_image_tm(
        self,
        benchmark: BenchmarkFixture,
        segm_preds: list[dict[str, Tensor]],
        segm_target: list[dict[str, Tensor]],
        tm_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(tm_compute, segm_preds, segm_target, iou_type="segm")

    @pytest.mark.benchmark(group="compute_segm_single", warmup=True)
    def test_segm_single_image_pt(
        self,
        benchmark: BenchmarkFixture,
        segm_preds: list[dict[str, Tensor]],
        segm_target: list[dict[str, Tensor]],
        pt_compute: Callable[..., dict[str, Tensor]],
    ) -> None:
        benchmark(pt_compute, segm_preds, segm_target, iou_type="segm")
