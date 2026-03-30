"""Test bbNms (bounding box NMS) with profiling."""

from _pytest.terminal import TerminalReporter
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import Tensor, profiler
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.mask_api.bb_nms import bbNms
from pytorchcocotools.utils.callable import resolve_actual_function


class BbNmsCases:
    """Test cases for bounding box NMS."""

    def case_no_suppression_high_threshold(self) -> tuple[Tensor, float, list[bool]]:
        """High threshold - no suppression."""
        dt = torch.tensor([[10, 10, 30, 30], [40, 40, 60, 60]], dtype=torch.float32)
        thr = 0.9
        expected = [True, True]
        return (dt, thr, expected)

    def case_suppress_overlapping(self) -> tuple[Tensor, float, list[bool]]:
        """Suppression of highly overlapping boxes."""
        dt = torch.tensor([[10, 10, 30, 30], [12, 12, 32, 32], [40, 40, 60, 60]], dtype=torch.float32)
        thr = 0.3
        expected = [True, False, True]  # Second box should be suppressed
        return (dt, thr, expected)

    def case_low_threshold_suppress_all(self) -> tuple[Tensor, float, list[bool]]:
        """Very low threshold suppresses many boxes."""
        dt = torch.tensor([[10, 10, 30, 30], [15, 15, 35, 35], [20, 20, 40, 40]], dtype=torch.float32)
        thr = 0.01
        expected = [True, False, False]  # Last two should be suppressed
        return (dt, thr, expected)

    def case_single_box(self) -> tuple[Tensor, float, list[bool]]:
        """Single box - no suppression."""
        dt = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
        thr = 0.5
        expected = [True]
        return (dt, thr, expected)

    def case_non_overlapping_boxes(self) -> tuple[Tensor, float, list[bool]]:
        """Non-overlapping boxes - no suppression."""
        dt = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]], dtype=torch.float32)
        thr = 0.5
        expected = [True, True, True]
        return (dt, thr, expected)


@pytest.mark.benchmark(group="bbNms", warmup=True)
@parametrize_with_cases("dt, thr, expected", cases=BbNmsCases)
def test_bb_nms_pt(
    benchmark: BenchmarkFixture,
    device: str,
    dt: Tensor,
    thr: float,
    expected: list[bool],
) -> None:
    """Test PyTorch implementation of bbNms."""
    dt_boxes = tv.BoundingBoxes(dt, format="XYXY", canvas_size=(100, 100), device=device)  # ty:ignore[no-matching-overload]

    result = benchmark(bbNms, dt_boxes, thr)

    assert result == expected


@pytest.mark.profiling
@parametrize_with_cases("dt, thr, expected", cases=BbNmsCases)
def test_bb_nms_pt_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    dt: Tensor,
    thr: float,
    expected: list[bool],
) -> None:
    """Profile PyTorch implementation of bbNms using torch.profiler."""
    dt_boxes = tv.BoundingBoxes(dt, format="XYXY", canvas_size=(100, 100), device=device)  # ty:ignore[no-matching-overload]

    # Warmup
    for _ in range(5):
        _ = bbNms(dt_boxes, thr)

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
        with_flops=True,
    ) as prof:
        for _ in range(10):
            _ = bbNms(dt_boxes, thr)

    # Print profiling results
    sort_by = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    terminal_writer.write_line(f"\n\n=== bbNms Profiling Results (device={device}) ===")
    terminal_writer.write_line(prof.key_averages().table(sort_by=sort_by, row_limit=10))


@pytest.mark.line_profiling
@parametrize_with_cases("dt, thr, expected", cases=BbNmsCases)
def test_bb_nms_pt_line_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    dt: Tensor,
    thr: float,
    expected: list[bool],
) -> None:
    """Profile PyTorch implementation of bbNms using line_profiler."""
    from line_profiler import LineProfiler

    dt_boxes = tv.BoundingBoxes(dt, format="XYXY", canvas_size=(100, 100), device=device)  # ty:ignore[no-matching-overload]

    # Warmup
    for _ in range(5):
        _ = bbNms(dt_boxes, thr)

    # Line profile
    lp = LineProfiler()
    target = resolve_actual_function(bbNms)
    lp.add_function(target)

    for _ in range(10):
        _ = lp.runcall(bbNms, dt_boxes, thr)

    # Print line profiling results
    terminal_writer.write_line(f"\n\n=== bbNms Line Profiling Results (device={device}) ===")
    lp.print_stats(output_unit=1e-6)
