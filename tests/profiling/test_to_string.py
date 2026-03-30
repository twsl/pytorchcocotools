"""Test rleToString (convert RLE to compressed string) with profiling."""

from _pytest.terminal import TerminalReporter
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import profiler
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RleObj
from pytorchcocotools.internal.mask_api.rle_encode import rleEncode
from pytorchcocotools.internal.mask_api.rle_fr_string import rleFrString
from pytorchcocotools.internal.mask_api.rle_to_string import rleToString
from pytorchcocotools.utils.callable import resolve_actual_function


class RleToStringCases:
    """Test cases for converting RLE to string."""

    h = 25
    w = 25

    def _create_mask(self, x1: int, y1: int, x2: int, y2: int) -> tv.Mask:
        """Create a mask from bounding box."""
        mask = torch.zeros((self.h, self.w), dtype=torch.uint8)
        mask[y1:y2, x1:x2] = 1
        return tv.Mask(mask.unsqueeze(0))

    def case_corner_area(self) -> tuple[RLE, bytes]:
        """Corner area."""
        masks = self._create_mask(0, 0, 5, 5)
        rle = rleEncode(masks)[0]
        expected = b"05d00000000d?"
        return (rle, expected)

    def case_center_area(self) -> tuple[RLE, bytes]:
        """Center area."""
        masks = self._create_mask(5, 5, 15, 15)
        rle = rleEncode(masks)[0]
        expected = b"R4:?00000000000000000e7"
        return (rle, expected)

    def case_end_area(self) -> tuple[RLE, bytes]:
        """End area."""
        masks = self._create_mask(20, 20, 25, 25)
        rle = rleEncode(masks)[0]
        expected = b"X`05d00000000"
        return (rle, expected)

    def case_full_area(self) -> tuple[RLE, bytes]:
        """Full area."""
        masks = self._create_mask(0, 0, 25, 25)
        rle = rleEncode(masks)[0]
        expected = b"0ac0"
        return (rle, expected)

    def case_from_rle_obj(self) -> tuple[RLE, bytes]:
        """From existing RLE object."""
        rle = rleFrString(b"T8:?00000000000000000c3", self.h, self.w)
        expected = b"T8:?00000000000000000c3"
        return (rle, expected)


@pytest.mark.benchmark(group="rleToString", warmup=True)
@parametrize_with_cases("rle, expected", cases=RleToStringCases)
def test_rle_to_string_pt(
    benchmark: BenchmarkFixture,
    device: str,
    rle: RLE,
    expected: bytes,
) -> None:
    """Test PyTorch implementation of rleToString."""
    result = benchmark(rleToString, rle)

    assert result == expected


@pytest.mark.profiling
@parametrize_with_cases("rle, expected", cases=RleToStringCases)
def test_rle_to_string_pt_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    rle: RLE,
    expected: bytes,
) -> None:
    """Profile PyTorch implementation of rleToString using torch.profiler."""
    # Warmup
    for _ in range(5):
        _ = rleToString(rle)

    # Profile
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(10):
            _ = rleToString(rle)

    # Print profiling results
    sort_by = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    terminal_writer.write_line(f"\n\n=== rleToString Profiling Results (device={device}) ===")
    terminal_writer.write_line(prof.key_averages().table(sort_by=sort_by, row_limit=10))


@pytest.mark.line_profiling
@parametrize_with_cases("rle, expected", cases=RleToStringCases)
def test_rle_to_string_pt_line_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    rle: RLE,
    expected: bytes,
) -> None:
    """Profile PyTorch implementation of rleToString using line_profiler."""
    from line_profiler import LineProfiler

    # Warmup
    for _ in range(5):
        _ = rleToString(rle)

    # Line profile
    lp = LineProfiler()
    target = resolve_actual_function(rleToString)
    lp.add_function(target)

    for _ in range(10):
        _ = lp.runcall(rleToString, rle)

    # Print line profiling results
    terminal_writer.write_line(f"\n\n=== rleToString Line Profiling Results (device={device}) ===")
    lp.print_stats(output_unit=1e-6)
