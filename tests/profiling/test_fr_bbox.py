"""Test rleFrBbox (convert bounding boxes to RLE) with profiling."""

from typing import cast

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torch import Tensor, profiler
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RleObj
from pytorchcocotools.internal.mask_api import rleToString
from pytorchcocotools.internal.mask_api.rle_fr_bbox import rleFrBbox


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
        expected = RleObj(size=[self.h, self.w], counts=b"")
        return (self.h, self.w, bbox, expected)

    def case_corner_box(self) -> tuple[int, int, Tensor, RleObj]:
        """Corner bounding box."""
        bbox = torch.tensor([[0, 0, 5, 5]], dtype=torch.float64)
        expected = RleObj(size=[self.h, self.w], counts=b"d05d000000d?")
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
    ) as prof:
        for _ in range(10):
            _ = rleFrBbox(bbox_tv)

    # Print profiling results
    print(f"\n\n=== rleFrBbox Profiling Results (device={device}) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
