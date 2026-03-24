"""Test rleToString (convert RLE to compressed string) with profiling."""

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
    ) as prof:
        for _ in range(10):
            _ = rleToString(rle)

    # Print profiling results
    print(f"\n\n=== rleToString Profiling Results (device={device}) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
