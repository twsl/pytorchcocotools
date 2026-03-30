"""Test rleMerge with profiling and compile diagnostics."""

import timeit
from typing import cast

from _pytest.terminal import TerminalReporter
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
import torch
from torch import profiler
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE
from pytorchcocotools.internal.mask_api.rle_encode import rleEncode
from pytorchcocotools.internal.mask_api.rle_fr_string import rleFrString
from pytorchcocotools.internal.mask_api.rle_merge import (
    _pack_rle_counts,
    _rle_merge_events_buffer,
    _rle_merge_events_buffer_eager,
    _trim_run_buffer,
    rleMerge,
)
from pytorchcocotools.internal.mask_api.rle_to_string import rleToString

COMPLEX_CANVAS = (427, 640)
COMPLEX_INPUTS = (
    b"\\`_3;j<6M3E_OjCd0T<:O1O2O001O00001O00001O001O0000O1K6J5J6A^C0g<N=O001O0O2Omk^4",
    b"RT_32n<<O100O0010O000010O0001O00001O000O101O0ISPc4",
)
COMPLEX_UNION = (
    b"RT_32X<9SD3f;3ZDNb;5_DKU;DeDd05HU;b0kD_OS;b0nD]OQ;e0nD[OR;e0nD[OR;f0nDZOQ;f0oDZOQ;"
    b"f0oDZOQ;g0oDXOQ;h0oDXOQ;h0oDXOQ;i0oDVOQ;j0oDVOQ;k0nDTOS;l0mDTOS;l0nDSOR;l0oDmNX;"
    b"n0f0J5J6A^C0g<N=O001O0O2Omk^4"
)

CASE_SPECS: dict[str, dict[str, object]] = {
    "overlap_intersection": {
        "kind": "squares",
        "canvas_size": (25, 25),
        "payload": ((5, 10), (8, 13)),
        "intersect": True,
        "expected": b"`62g00_;",
    },
    "complex_pair_union": {
        "kind": "strings",
        "canvas_size": COMPLEX_CANVAS,
        "payload": COMPLEX_INPUTS,
        "intersect": False,
        "expected": COMPLEX_UNION,
    },
    "complex_multi_union": {
        "kind": "strings",
        "canvas_size": COMPLEX_CANVAS,
        "payload": COMPLEX_INPUTS * 4,
        "intersect": False,
        "expected": COMPLEX_UNION,
    },
}

BENCHMARK_CASES = tuple(CASE_SPECS)
PROFILE_CASES = ("complex_pair_union", "complex_multi_union")


def _build_square_rle(device: str, h: int, w: int, start: int, end: int) -> RLE:
    mask = torch.zeros((1, h, w), dtype=torch.uint8, device=device)
    mask[0, start:end, start:end] = 1
    return rleEncode(tv.Mask(mask, device=device))[0]


def _build_case(case_name: str, device: str) -> tuple[list[RLE], bool, bytes]:
    spec = CASE_SPECS[case_name]
    h, w = cast(tuple[int, int], spec["canvas_size"])
    intersect = cast(bool, spec["intersect"])
    expected = cast(bytes, spec["expected"])

    if spec["kind"] == "strings":
        payload = cast(tuple[bytes, ...], spec["payload"])
        rles = [rleFrString(counts, h, w, device=device) for counts in payload]
    else:
        payload = cast(tuple[tuple[int, int], ...], spec["payload"])
        rles = [_build_square_rle(device, h, w, start, end) for start, end in payload]

    return rles, intersect, expected


@pytest.mark.benchmark(group="rleMerge", warmup=True)
@pytest.mark.parametrize("case_name", BENCHMARK_CASES, ids=BENCHMARK_CASES)
def test_rle_merge_pt(
    benchmark: BenchmarkFixture,
    device: str,
    case_name: str,
) -> None:
    """Benchmark the internal RLE merge implementation."""
    rles, intersect, expected = _build_case(case_name, device)

    result = cast(RLE, benchmark(rleMerge, rles, intersect))

    assert result.cnts.device.type == device
    assert rleToString(result) == expected


@pytest.mark.profiling
@pytest.mark.parametrize("case_name", PROFILE_CASES, ids=PROFILE_CASES)
def test_rle_merge_pt_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    case_name: str,
) -> None:
    """Profile the internal RLE merge implementation using torch.profiler."""
    rles, intersect, expected = _build_case(case_name, device)

    for _ in range(5):
        _ = rleMerge(rles, intersect)

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
        if device == "cuda"
        else [profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(10):
            result = rleMerge(rles, intersect)

    assert rleToString(result) == expected

    sort_by = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    terminal_writer.write_line(f"\n\n=== rleMerge Profiling Results ({case_name}, device={device}) ===")
    terminal_writer.write_line(prof.key_averages().table(sort_by=sort_by, row_limit=10))


@pytest.mark.line_profiling
@pytest.mark.parametrize("case_name", PROFILE_CASES, ids=PROFILE_CASES)
def test_rle_merge_pt_line_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    case_name: str,
) -> None:
    """Profile the pure-PyTorch merge kernel using line_profiler."""
    from line_profiler import LineProfiler

    rles, intersect, expected = _build_case(case_name, device)
    packed = _pack_rle_counts(rles)
    assert packed is not None
    h, w, _, padded_counts, lengths, total_pixels = packed

    for _ in range(5):
        run_buffer, run_count = _rle_merge_events_buffer_eager(padded_counts, lengths, intersect, total_pixels)
        _ = _trim_run_buffer(run_buffer, run_count)

    lp = LineProfiler()
    lp.add_function(_pack_rle_counts)
    lp.add_function(_rle_merge_events_buffer_eager)
    lp.add_function(_trim_run_buffer)

    for _ in range(10):
        run_buffer, run_count = lp.runcall(
            _rle_merge_events_buffer_eager,
            padded_counts,
            lengths,
            intersect,
            total_pixels,
        )
        counts = _trim_run_buffer(run_buffer, run_count)
        result = RLE(h, w, counts)

    assert rleToString(result) == expected

    terminal_writer.write_line(f"\n\n=== rleMerge Line Profiling Results ({case_name}, device={device}) ===")
    lp.print_stats(output_unit=1e-6)
