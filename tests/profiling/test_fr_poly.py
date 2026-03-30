"""Test rleFrPoly (convert polygons to RLE) with profiling."""

from typing import cast

from _pytest.terminal import TerminalReporter
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import profiler

from pytorchcocotools.internal.entities import RLE, RleObj
from pytorchcocotools.internal.mask_api.rle_fr_poly import rleFrPoly
from pytorchcocotools.internal.mask_api.rle_to_string import rleToString
from pytorchcocotools.utils.callable import resolve_actual_function
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

    def case_real_small_polygon(self) -> tuple[int, int, Polygon, RleObj]:
        """Real-life small polygon (7 vertices) from data/example.json."""
        h, w = 427, 640
        flat = torch.tensor(
            [
                [
                    266.35,
                    214.44,
                    270.41,
                    217.3,
                    276.38,
                    218.97,
                    282.11,
                    218.97,
                    285.93,
                    217.3,
                    286.88,
                    207.28,
                    267.07,
                    201.07,
                ]
            ],
            dtype=torch.float64,
        )
        poly = Polygon(flat.reshape(-1, 2), canvas_size=(h, w))  # ty:ignore[no-matching-overload]
        expected = RleObj(
            size=[h, w],
            counts=b"RT_32n<<O100O0010O000010O0001O00001O000O101O0ISPc4",
        )
        return (h, w, poly, expected)

    def case_real_large_polygon(self) -> tuple[int, int, Polygon, RleObj]:
        """Real-life large polygon (53 vertices) from data/example.json."""
        h, w = 640, 480
        flat = torch.tensor(
            [
                [
                    135.17,
                    444.83,
                    113.1,
                    435.17,
                    100.69,
                    439.31,
                    96.55,
                    439.31,
                    75.86,
                    431.03,
                    71.72,
                    428.28,
                    70.34,
                    396.55,
                    64.83,
                    356.55,
                    89.66,
                    338.62,
                    104.83,
                    312.41,
                    126.9,
                    291.72,
                    143.45,
                    253.1,
                    144.83,
                    221.38,
                    146.21,
                    199.31,
                    157.24,
                    184.14,
                    172.41,
                    160.69,
                    182.07,
                    155.17,
                    200,
                    149.66,
                    219.31,
                    142.76,
                    234.48,
                    141.38,
                    266.21,
                    149.66,
                    277.24,
                    163.45,
                    284.14,
                    171.72,
                    297.93,
                    195.17,
                    307.59,
                    215.86,
                    317.24,
                    248.97,
                    320,
                    277.93,
                    318.62,
                    293.1,
                    321.38,
                    306.9,
                    342.07,
                    323.45,
                    360,
                    353.79,
                    365.52,
                    371.72,
                    364.14,
                    393.79,
                    354.48,
                    406.21,
                    342.07,
                    413.1,
                    324.14,
                    417.24,
                    303.45,
                    420,
                    303.45,
                    420,
                    303.45,
                    421.38,
                    296.55,
                    411.72,
                    288.28,
                    370.34,
                    286.9,
                    366.21,
                    245.52,
                    362.07,
                    219.31,
                    371.72,
                    206.9,
                    384.14,
                    194.48,
                    400.69,
                    175.17,
                    415.86,
                    172.41,
                    439.31,
                    168.28,
                    457.24,
                    155.86,
                    462.76,
                    143.45,
                    448.97,
                    133.79,
                    440.69,
                    129.66,
                    437.93,
                ]
            ],
            dtype=torch.float64,
        )
        poly = Polygon(flat.reshape(-1, 2), canvas_size=(h, w))  # ty:ignore[no-matching-overload]
        expected = RleObj(
            size=[h, w],
            counts=(
                b"ToX15jc09H7H8H8H:Gf0YO9G1O2O1N1O2O0O1O2N101N1O101N1O2N100N3M2O2M2N3M"
                b"2O1N2N2N2O1N2N2O0O2O1O001O1O001O1O001O2N1O2O0O1O2N1O2N1O2N1O2N1O1M4M2"
                b"MOPAYKm>j4SAUKk>n4UARKg>Q5[AmJc>V513L4M3M3L4M2N3L4K5XOh0]Oc0D<N2M4M2M3"
                b"N2N2M3N2M3N1O1N2O0O2N101N101N101N100ON2M4M2M5H7I8I610O001O01O01O1O00100"
                b"O001O1O10O01O1O001O1O0O2O1O001N2O001N2O001O1O001O1O001O001O1O001O1O1O100"
                b"0O0100O10000O10000O10O0101O0O10001N100O101O0O10001O00001O01O0001O0000001O"
                b"0001O01O0000001O1O1O2N100O1O2N1O1O1O2N1O1O100O2N1O1O2N2N1102N3L4M3M2M4M"
                b"3M3M3L10O100O2O0O1N2N2N2N2M5L3M4L3M4L3M4K4M3M6J:F;^KPA^3m?L5K1N2O001O1"
                b"O1N2O001O1O1N2O001O1N2O1O001O1N2O2M3N1N3N2M2N3N2M2O2M3N1N3M3L3N3M3M3L6"
                b"K4L4L4G9]OmlV2"
            ),
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
    terminal_writer: TerminalReporter,
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
        with_flops=True,
    ) as prof:
        for _ in range(10):
            _ = rleFrPoly(poly)

    # Print profiling results
    sort_by = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    terminal_writer.write_line(f"\n\n=== rleFrPoly Profiling Results (device={device}) ===")
    terminal_writer.write_line(prof.key_averages().table(sort_by=sort_by, row_limit=10))


@pytest.mark.line_profiling
@parametrize_with_cases("h, w, poly, expected", cases=RleFrPolyCases)
def test_rle_fr_poly_pt_line_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    h: int,
    w: int,
    poly: Polygon,
    expected: RleObj,
) -> None:
    """Profile PyTorch implementation of rleFrPoly using line_profiler."""
    from line_profiler import LineProfiler

    poly = poly.to(device)  # ty:ignore[invalid-assignment]

    # Warmup
    for _ in range(5):
        _ = rleFrPoly(poly)

    # Line profile
    lp = LineProfiler()
    target = resolve_actual_function(rleFrPoly)
    lp.add_function(target)

    for _ in range(10):
        _ = lp.runcall(rleFrPoly, poly)

    # Print line profiling results
    terminal_writer.write_line(f"\n\n=== rleFrPoly Line Profiling Results (device={device}) ===")
    lp.print_stats(output_unit=1e-6)
