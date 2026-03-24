"""Test rleFrString (convert compressed string to RLE) with profiling."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import profiler

from pytorchcocotools.internal.entities import RleObj
from pytorchcocotools.internal.mask_api.rle_fr_string import rleFrString
from pytorchcocotools.internal.mask_api.rle_to_string import rleToString


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
    ) as prof:
        for _ in range(10):
            _ = rleFrString(s, h, w)

    # Print profiling results
    print(f"\n\n=== rleFrString Profiling Results (device={device}) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
