"""Test rleFrPoly (convert polygons to RLE) with profiling."""

from typing import cast

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import profiler

from pytorchcocotools.internal.entities import RLE, RleObj
from pytorchcocotools.internal.mask_api.rle_fr_poly import rleFrPoly
from pytorchcocotools.internal.mask_api.rle_to_string import rleToString
from pytorchcocotools.utils.poly import Polygon


class RleFrPolyCases:
    """Test cases for converting polygons to RLE."""

    h = 25
    w = 25

    def case_simple_polygon(self) -> tuple[int, int, Polygon, RleObj]:
        """Simple polygon."""
        flat = torch.tensor([[10, 10, 20, 10, 20, 20, 21, 21, 10, 20]], dtype=torch.float64)
        poly = Polygon(flat.reshape(-1, 2), canvas_size=(self.h, self.w))  # ty:ignore[no-matching-overload]
        expected = RleObj(size=[self.h, self.w], counts=b"T8:?00000000001O00000:F`2")
        return (self.h, self.w, poly, expected)

    def case_triangle(self) -> tuple[int, int, Polygon, RleObj]:
        """Triangle polygon."""
        flat = torch.tensor([[10, 10, 20, 10, 15, 20]], dtype=torch.float64)
        poly = Polygon(flat.reshape(-1, 2), canvas_size=(self.h, self.w))  # ty:ignore[no-matching-overload]
        expected = RleObj(
            size=[self.h, self.w],
            counts=b"T81h02N2N2N2N00N2N2N2Ne3",
        )
        return (self.h, self.w, poly, expected)

    def case_rectangle_polygon(self) -> tuple[int, int, Polygon, RleObj]:
        """Rectangle as polygon."""
        flat = torch.tensor([[5, 5, 15, 5, 15, 15, 5, 15]], dtype=torch.float64)
        poly = Polygon(flat.reshape(-1, 2), canvas_size=(self.h, self.w))  # ty:ignore[no-matching-overload]
        expected = RleObj(size=[self.h, self.w], counts=b"R4:?00000000000000000e7")
        return (self.h, self.w, poly, expected)

    def case_complex_polygon(self) -> tuple[int, int, Polygon, RleObj]:
        """More complex polygon."""
        h, w = 427, 640
        flat = torch.tensor(
            [
                [
                    266.83,
                    189.37,
                    267.79,
                    175.29,
                    269.46,
                    170.04,
                    271.37,
                    165.98,
                    270.89,
                    163.12,
                    269.12,
                    159.54,
                    272.8,
                    156.44,
                    287.36,
                    156.44,
                    293.33,
                    157.87,
                    296.91,
                    160.49,
                    296.91,
                    161.21,
                    291.89,
                    161.92,
                    289.98,
                    165.03,
                    291.42,
                    169.56,
                    285.16,
                    196.54,
                ]
            ],
            dtype=torch.float64,
        )
        poly = Polygon(flat.reshape(-1, 2), canvas_size=(h, w))  # ty:ignore[no-matching-overload]
        expected = RleObj(
            size=[h, w],
            counts=b"\\`_3;j<6B@nCc0Q<@kCc0S<;01N10001O001O00001O001O0000O1L4K6K4L4B]COh<O<O001O0O2Omk^4",
        )
        return (h, w, poly, expected)


@pytest.mark.benchmark(group="rleFrPoly", warmup=True)
@parametrize_with_cases("h, w, poly, expected", cases=RleFrPolyCases)
def test_rle_fr_poly_pt(
    benchmark: BenchmarkFixture,
    device: str,
    h: int,
    w: int,
    poly: Polygon,
    expected: RleObj,
) -> None:
    """Test PyTorch implementation of rleFrPoly."""
    poly = poly.to(device)  # ty:ignore[invalid-assignment]

    rle = cast(RLE, benchmark(rleFrPoly, poly))

    assert [rle.h, rle.w] == expected["size"]
    assert rleToString(rle) == expected["counts"]


@pytest.mark.profiling
@parametrize_with_cases("h, w, poly, expected", cases=RleFrPolyCases)
def test_rle_fr_poly_pt_profiling(
    device: str,
    h: int,
    w: int,
    poly: Polygon,
    expected: RleObj,
) -> None:
    """Profile PyTorch implementation of rleFrPoly using torch.profiler."""
    poly = poly.to(device)  # ty:ignore[invalid-assignment]

    # Warmup
    for _ in range(5):
        _ = rleFrPoly(poly)

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
            _ = rleFrPoly(poly)

    # Print profiling results
    print(f"\n\n=== rleFrPoly Profiling Results (device={device}) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
