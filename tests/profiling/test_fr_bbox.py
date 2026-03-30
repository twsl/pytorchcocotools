"""Test rleFrBbox (convert bounding boxes to RLE) with profiling."""

from typing import cast

from _pytest.terminal import TerminalReporter
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import Tensor, profiler
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RleObj
from pytorchcocotools.internal.mask_api import rleToString
from pytorchcocotools.internal.mask_api.rle_fr_bbox import rleFrBbox
from pytorchcocotools.utils.callable import resolve_actual_function


class RleFrBboxCases:
    """Test cases for converting bounding boxes to RLE."""

    h = 25
    w = 25

    def case_single_box(self) -> tuple[int, int, Tensor, RleObj]:
        """Single bounding box."""
        bbox = torch.tensor([[10, 10, 10, 10]], dtype=torch.float64)
        expected = RleObj(size=[self.h, self.w], counts=b"T8:?00000000000000000c3")
        return (self.h, self.w, bbox, expected)

    def case_multiple_boxes(self) -> tuple[int, int, Tensor, list[RleObj]]:
        """Multiple bounding boxes."""
        bboxes = torch.tensor([[5, 5, 10, 10], [15, 15, 5, 5]], dtype=torch.float64)
        expected = [
            RleObj(size=[self.h, self.w], counts=b"R4:?00000000000000000e7"),
            RleObj(size=[self.h, self.w], counts=b"V<5d00000000^3"),
        ]
        return (self.h, self.w, bboxes, expected)

    def case_full_box(self) -> tuple[int, int, Tensor, RleObj]:
        """Full canvas bounding box."""
        bbox = torch.tensor([[0, 0, self.w, self.h]], dtype=torch.float64)
        expected = RleObj(size=[self.h, self.w], counts=b"0ac0")
        return (self.h, self.w, bbox, expected)

    def case_corner_box(self) -> tuple[int, int, Tensor, RleObj]:
        """Corner bounding box."""
        bbox = torch.tensor([[0, 0, 5, 5]], dtype=torch.float64)
        expected = RleObj(size=[self.h, self.w], counts=b"05d00000000d?")
        return (self.h, self.w, bbox, expected)


@pytest.mark.benchmark(group="rleFrBbox", warmup=True)
@parametrize_with_cases("h, w, bbox, expected", cases=RleFrBboxCases)
def test_rle_fr_bbox_pt(
    benchmark: BenchmarkFixture,
    device: str,
    h: int,
    w: int,
    bbox: Tensor,
    expected: RleObj | list[RleObj],
) -> None:
    """Test PyTorch implementation of rleFrBbox."""
    bbox_tv = tv.BoundingBoxes(bbox, format="XYWH", canvas_size=(h, w), device=device)  # ty:ignore[no-matching-overload]

    result = cast(list[RLE], benchmark(rleFrBbox, bbox_tv))

    # Convert result to RleObj for easier comparison with expected.
    objs = [
        RleObj(
            size=[r.h, r.w],
            counts=rleToString(
                r,
                device=device,
            ),
        )
        for r in result
    ]

    if not isinstance(expected, list):
        expected = [expected]

    assert len(result) == len(expected)
    for r, e in zip(objs, expected, strict=False):
        assert r.size == e.size
        assert r.counts == e.counts


@pytest.mark.profiling
@parametrize_with_cases("h, w, bbox, expected", cases=RleFrBboxCases)
def test_rle_fr_bbox_pt_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    h: int,
    w: int,
    bbox: Tensor,
    expected: RleObj | list[RleObj],
) -> None:
    """Profile PyTorch implementation of rleFrBbox using torch.profiler."""
    bbox_tv = tv.BoundingBoxes(bbox, format="XYWH", canvas_size=(h, w), device=device)  # ty:ignore[no-matching-overload]

    # Warmup
    for _ in range(5):
        _ = rleFrBbox(bbox_tv)

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
            _ = rleFrBbox(bbox_tv)

    # Print profiling results
    sort_by = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    terminal_writer.write_line(f"\n\n=== rleFrBbox Profiling Results (device={device}) ===")
    terminal_writer.write_line(prof.key_averages().table(sort_by=sort_by, row_limit=10))


@pytest.mark.line_profiling
@parametrize_with_cases("h, w, bbox, expected", cases=RleFrBboxCases)
def test_rle_fr_bbox_pt_line_profiling(
    terminal_writer: TerminalReporter,
    device: str,
    h: int,
    w: int,
    bbox: Tensor,
    expected: RleObj | list[RleObj],
) -> None:
    """Profile PyTorch implementation of rleFrBbox using line_profiler."""
    from line_profiler import LineProfiler

    bbox_tv = tv.BoundingBoxes(bbox, format="XYWH", canvas_size=(h, w), device=device)  # ty:ignore[no-matching-overload]

    # Warmup
    for _ in range(5):
        _ = rleFrBbox(bbox_tv)

    # Line profile
    lp = LineProfiler()
    target = resolve_actual_function(rleFrBbox)
    lp.add_function(target)

    for _ in range(10):
        _ = lp.runcall(rleFrBbox, bbox_tv)

    # Print line profiling results
    terminal_writer.write_line(f"\n\n=== rleFrBbox Line Profiling Results (device={device}) ===")
    lp.print_stats(output_unit=1e-6)
