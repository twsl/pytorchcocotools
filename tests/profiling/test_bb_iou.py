"""Test bbIou (bounding box IoU) with profiling."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import Tensor, profiler
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.mask_api.bb_iou import bbIou


class BbIouCases:
    """Test cases for bounding box IoU."""

    def case_identical_boxes(self) -> tuple[Tensor, Tensor, list[bool], Tensor]:
        """Two identical boxes should have IoU of 1.0."""
        dt = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
        gt = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
        iscrowd = [False]
        expected = torch.tensor([[1.0]], dtype=torch.float32)
        return (dt, gt, iscrowd, expected)

    def case_non_overlapping_boxes(self) -> tuple[Tensor, Tensor, list[bool], Tensor]:
        """Non-overlapping boxes should have IoU of 0.0."""
        dt = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
        gt = torch.tensor([[40, 40, 60, 60]], dtype=torch.float32)
        iscrowd = [False]
        expected = torch.tensor([[0.0]], dtype=torch.float32)
        return (dt, gt, iscrowd, expected)

    def case_partial_overlap(self) -> tuple[Tensor, Tensor, list[bool], Tensor]:
        """Partially overlapping boxes."""
        dt = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
        gt = torch.tensor([[20, 20, 40, 40]], dtype=torch.float32)
        iscrowd = [False]
        # Intersection: 100, Union: 700 -> IoU = 1/7
        expected = torch.tensor([[1.0 / 7.0]], dtype=torch.float32)
        return (dt, gt, iscrowd, expected)

    def case_crowd_annotation(self) -> tuple[Tensor, Tensor, list[bool], Tensor]:
        """Crowd annotations use different IoU calculation."""
        dt = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
        gt = torch.tensor([[20, 20, 40, 40]], dtype=torch.float32)
        iscrowd = [True]
        # For crowd: IoU = intersection / area(dt)
        expected = torch.tensor([[0.25]], dtype=torch.float32)
        return (dt, gt, iscrowd, expected)

    def case_multiple_boxes(self) -> tuple[Tensor, Tensor, list[bool], Tensor]:
        """Multiple detections and ground truths."""
        dt = torch.tensor([[10, 10, 30, 30], [40, 40, 60, 60]], dtype=torch.float32)
        gt = torch.tensor([[10, 10, 30, 30], [50, 50, 70, 70]], dtype=torch.float32)
        iscrowd = [False, False]
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0 / 7.0]], dtype=torch.float32)
        return (dt, gt, iscrowd, expected)


@pytest.mark.benchmark(group="bbIou", warmup=True)
@parametrize_with_cases("dt, gt, iscrowd, expected", cases=BbIouCases)
def test_bb_iou_pt(
    benchmark: BenchmarkFixture,
    device: str,
    dt: Tensor,
    gt: Tensor,
    iscrowd: list[bool],
    expected: Tensor,
) -> None:
    """Test PyTorch implementation of bbIou."""
    # Convert to tv_tensors
    dt_boxes = tv.BoundingBoxes(dt, format="XYXY", canvas_size=(100, 100), device=device)  # ty:ignore[no-matching-overload]
    gt_boxes = tv.BoundingBoxes(gt, format="XYXY", canvas_size=(100, 100), device=device)  # ty:ignore[no-matching-overload]

    # Compute IoU
    result = benchmark(bbIou, dt_boxes, gt_boxes, iscrowd)

    # Compare results
    assert result.device.type == device
    torch.testing.assert_close(result, expected.to(device), rtol=1e-5, atol=1e-5)


@pytest.mark.profiling
@parametrize_with_cases("dt, gt, iscrowd, expected", cases=BbIouCases)
def test_bb_iou_pt_profiling(
    device: str,
    dt: Tensor,
    gt: Tensor,
    iscrowd: list[bool],
    expected: Tensor,
) -> None:
    """Profile PyTorch implementation of bbIou using torch.profiler."""
    dt_boxes = tv.BoundingBoxes(dt, format="XYXY", canvas_size=(100, 100), device=device)  # ty:ignore[no-matching-overload]
    gt_boxes = tv.BoundingBoxes(gt, format="XYXY", canvas_size=(100, 100), device=device)  # ty:ignore[no-matching-overload]

    # Warmup
    for _ in range(5):
        _ = bbIou(dt_boxes, gt_boxes, iscrowd)

    # Profile
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ]
        if device == "cuda"
        else [profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            result = bbIou(dt_boxes, gt_boxes, iscrowd)

    # Verify correctness
    torch.testing.assert_close(result, expected.to(device), rtol=1e-5, atol=1e-5)

    # Print profiling results
    print(f"\n\n=== bbIou Profiling Results (device={device}) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
