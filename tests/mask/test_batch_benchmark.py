"""Benchmark comparisons for batched vs non-batched operations."""
import pytest
import torch
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs
from pytorchcocotools.internal.mask_api import (
    bbNms,
    bbNmsBatch,
    rleArea,
    rleAreaBatch,
    rleNms,
    rleNmsBatch,
)


@pytest.mark.benchmark(group="area")
def test_rle_area_original(benchmark, device: str):
    """Benchmark original rleArea."""
    # Create RLE masks
    rles = RLEs([
        RLE(100, 100, torch.tensor([0, 25, 75, 0], device=device))
        for _ in range(50)
    ])
    
    result = benchmark(rleArea, rles)
    assert len(result) == 50


@pytest.mark.benchmark(group="area")
def test_rle_area_batch(benchmark, device: str):
    """Benchmark batched rleAreaBatch."""
    # Create RLE masks
    rles = RLEs([
        RLE(100, 100, torch.tensor([0, 25, 75, 0], device=device))
        for _ in range(50)
    ])
    
    result = benchmark(rleAreaBatch, rles)
    assert result.shape == (50,)


@pytest.mark.benchmark(group="bb_nms")
def test_bb_nms_original(benchmark, device: str):
    """Benchmark original bbNms."""
    dt = tv.BoundingBoxes(
        torch.rand(20, 4, device=device) * 100,
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(200, 200),
    )
    
    result = benchmark(bbNms, dt, 0.5)
    assert isinstance(result, list)


@pytest.mark.benchmark(group="bb_nms")
def test_bb_nms_batch(benchmark, device: str):
    """Benchmark batched bbNmsBatch."""
    dt = tv.BoundingBoxes(
        torch.rand(20, 4, device=device) * 100,
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(200, 200),
    )
    
    result = benchmark(bbNmsBatch, dt, 0.5)
    assert result.dtype == torch.bool


@pytest.mark.benchmark(group="rle_nms")
def test_rle_nms_original(benchmark, device: str):
    """Benchmark original rleNms."""
    # Create simple RLE masks
    rles = RLEs([
        RLE(50, 50, torch.tensor([0, 100, 2400, 0], device=device))
        for _ in range(10)
    ])
    
    result = benchmark(rleNms, rles, 10, 0.5)
    assert isinstance(result, list)


@pytest.mark.benchmark(group="rle_nms")
def test_rle_nms_batch(benchmark, device: str):
    """Benchmark batched rleNmsBatch."""
    # Create simple RLE masks
    rles = RLEs([
        RLE(50, 50, torch.tensor([0, 100, 2400, 0], device=device))
        for _ in range(10)
    ])
    
    result = benchmark(rleNmsBatch, rles, 10, 0.5)
    assert result.dtype == torch.bool
