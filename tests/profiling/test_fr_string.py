"""Test rleFrString (convert compressed string to RLE) with profiling."""

from _pytest.terminal import TerminalReporter
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import profiler

from pytorchcocotools.internal.entities import RleObj
from pytorchcocotools.internal.mask_api.rle_fr_string import rleFrString
from pytorchcocotools.internal.mask_api.rle_to_string import rleToString
from pytorchcocotools.utils.callable import resolve_actual_function


class RleFrStringCases:
    """Test cases for converting compressed strings to RLE."""

    h = 25
    w = 25

    def case_simple_rle(self) -> tuple[bytes, int, int, RleObj]:
        """Simple RLE string."""
        s = b"T8:?00000000000000000c3"
        expected = RleObj(size=[self.h, self.w], counts=b"T8:?00000000000000000c3")
        return (s, self.h, self.w, expected)

    def case_center_area(self) -> tuple[bytes, int, int, RleObj]:
        """Center area RLE string."""
        s = b"R45d00000000b;"
        expected = RleObj(size=[self.h, self.w], counts=b"R45d00000000b;")
        return (s, self.h, self.w, expected)

    def case_full_area(self) -> tuple[bytes, int, int, RleObj]:
        """Full area RLE string."""
        s = b"0ac0"
        expected = RleObj(size=[self.h, self.w], counts=b"0ac0")
        return (s, self.h, self.w, expected)

    def case_complex_rle(self) -> tuple[bytes, int, int, RleObj]:
        """Complex RLE string."""
        h, w = 427, 640
        s = b"\\`_3;j<6B@nCc0Q<@kCc0S<;01N10001O001O00001O001O0000O1L4K6K4L4B]COh<O<O001O0O2Omk^4"
        expected = RleObj(size=[h, w], counts=s)
        return (s, h, w, expected)

    def case_another_complex(self) -> tuple[bytes, int, int, RleObj]:
        """Another complex RLE string."""
        h, w = 427, 640
        s = b"RT_32n<<O100O0010O000010O0001O00001O000O101O0ISPc4"
        expected = RleObj(size=[h, w], counts=s)
        return (s, h, w, expected)


@pytest.mark.benchmark(group="rleFrString", warmup=True)
@parametrize_with_cases("s, h, w, expected", cases=RleFrStringCases)
def test_rle_fr_string_pt(
    benchmark: BenchmarkFixture,
    device: str,
    s: bytes,
    h: int,
    w: int,
    expected: RleObj,
) -> None:
    """Test PyTorch implementation of rleFrString."""
    result = benchmark(rleFrString, s, h, w)

    assert [result.h, result.w] == expected["size"]
    assert rleToString(result) == expected["counts"]


@pytest.mark.profiling
@parametrize_with_cases("s, h, w, expected", cases=RleFrStringCases)
def test_rle_fr_string_pt_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    s: bytes,
    h: int,
    w: int,
    expected: RleObj,
) -> None:
    """Profile PyTorch implementation of rleFrString using torch.profiler."""
    # Warmup
    for _ in range(5):
        _ = rleFrString(s, h, w)

    # Profile
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(10):
            _ = rleFrString(s, h, w)

    # Print profiling results
    sort_by = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    terminal_writer.write_line(f"\n\n=== rleFrString Profiling Results (device={device}) ===")
    terminal_writer.write_line(prof.key_averages().table(sort_by=sort_by, row_limit=10))


@pytest.mark.line_profiling
@parametrize_with_cases("s, h, w, expected", cases=RleFrStringCases)
def test_rle_fr_string_pt_line_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    s: bytes,
    h: int,
    w: int,
    expected: RleObj,
) -> None:
    """Profile PyTorch implementation of rleFrString using line_profiler."""
    from line_profiler import LineProfiler

    # Warmup
    for _ in range(5):
        _ = rleFrString(s, h, w)

    # Line profile
    lp = LineProfiler()
    target = resolve_actual_function(rleFrString)
    lp.add_function(target)

    for _ in range(10):
        _ = lp.runcall(rleFrString, s, h, w)

    # Print line profiling results
    terminal_writer.write_line(f"\n\n=== rleFrString Line Profiling Results (device={device}) ===")
    lp.print_stats(output_unit=1e-6)
