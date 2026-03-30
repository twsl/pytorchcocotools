"""Test rleNms (RLE NMS) with profiling."""

from _pytest.terminal import TerminalReporter
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import profiler
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLEs
from pytorchcocotools.internal.mask_api.rle_encode import rleEncode
from pytorchcocotools.internal.mask_api.rle_nms import rleNms
from pytorchcocotools.utils.callable import resolve_actual_function


class RleNmsCases:
    """Test cases for RLE NMS."""

    h = 50
    w = 50

    def _create_masks(self, *boxes: tuple[int, int, int, int]) -> tv.Mask:
        """Create masks from bounding boxes."""
        masks = []
        for x1, y1, x2, y2 in boxes:
            mask = torch.zeros((self.h, self.w), dtype=torch.uint8)
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)
        return tv.Mask(torch.stack(masks))

    def case_no_suppression_high_threshold(self) -> tuple[RLEs, int, float, list[bool]]:
        """High threshold - no suppression."""
        masks = self._create_masks((10, 10, 30, 30), (35, 35, 45, 45))
        rles = rleEncode(masks)
        # For NMS, we typically consider the first n detections with scores
        # Here we assume all are considered
        n = len(rles)
        thr = 0.9
        expected = [True, True]
        return (rles, n, thr, expected)

    def case_suppress_overlapping(self) -> tuple[RLEs, int, float, list[bool]]:
        """Suppression of highly overlapping masks."""
        masks = self._create_masks((10, 10, 30, 30), (12, 12, 32, 32), (35, 35, 45, 45))
        rles = rleEncode(masks)
        n = len(rles)
        thr = 0.3
        expected = [True, False, True]  # Second mask should be suppressed
        return (rles, n, thr, expected)

    def case_low_threshold_suppress_many(self) -> tuple[RLEs, int, float, list[bool]]:
        """Very low threshold suppresses many masks."""
        masks = self._create_masks((10, 10, 30, 30), (15, 15, 35, 35), (20, 20, 40, 40))
        rles = rleEncode(masks)
        n = len(rles)
        thr = 0.01
        expected = [True, False, False]  # Last two should be suppressed
        return (rles, n, thr, expected)

    def case_single_mask(self) -> tuple[RLEs, int, float, list[bool]]:
        """Single mask - no suppression."""
        masks = self._create_masks((10, 10, 30, 30))
        rles = rleEncode(masks)
        n = len(rles)
        thr = 0.5
        expected = [True]
        return (rles, n, thr, expected)

    def case_top_n_only(self) -> tuple[RLEs, int, float, list[bool]]:
        """Only consider top n detections."""
        masks = self._create_masks((10, 10, 20, 20), (12, 12, 22, 22), (30, 30, 40, 40), (32, 32, 42, 42))
        rles = rleEncode(masks)
        n = 2  # Only consider first 2
        thr = 0.3
        # Only first 2 are considered, second is suppressed
        expected = [True, False]
        return (rles, n, thr, expected)


@pytest.mark.benchmark(group="rleNms", warmup=True)
@parametrize_with_cases("rles, n, thr, expected", cases=RleNmsCases)
def test_rle_nms_pt(
    benchmark: BenchmarkFixture,
    device: str,
    rles: RLEs,
    n: int,
    thr: float,
    expected: list[bool],
) -> None:
    """Test PyTorch implementation of rleNms."""
    result = benchmark(rleNms, rles, n, thr)

    assert result == expected


@pytest.mark.profiling
@parametrize_with_cases("rles, n, thr, expected", cases=RleNmsCases)
def test_rle_nms_pt_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    rles: RLEs,
    n: int,
    thr: float,
    expected: list[bool],
) -> None:
    """Profile PyTorch implementation of rleNms using torch.profiler."""
    # Warmup
    for _ in range(5):
        _ = rleNms(rles, n, thr)

    # Profile
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(10):
            _ = rleNms(rles, n, thr)

    # Print profiling results
    sort_by = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    terminal_writer.write_line(f"\n\n=== rleNms Profiling Results (device={device}) ===")
    terminal_writer.write_line(prof.key_averages().table(sort_by=sort_by, row_limit=10))


@pytest.mark.line_profiling
@parametrize_with_cases("rles, n, thr, expected", cases=RleNmsCases)
def test_rle_nms_pt_line_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    rles: RLEs,
    n: int,
    thr: float,
    expected: list[bool],
) -> None:
    """Profile PyTorch implementation of rleNms using line_profiler."""
    from line_profiler import LineProfiler

    # Warmup
    for _ in range(5):
        _ = rleNms(rles, n, thr)

    # Line profile
    lp = LineProfiler()
    target = resolve_actual_function(rleNms)
    lp.add_function(target)

    for _ in range(10):
        _ = lp.runcall(rleNms, rles, n, thr)

    # Print line profiling results
    terminal_writer.write_line(f"\n\n=== rleNms Line Profiling Results (device={device}) ===")
    lp.print_stats(output_unit=1e-6)
