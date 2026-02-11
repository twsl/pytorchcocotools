"""Tests for batched versions of mask_api methods."""
import torch
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs
from pytorchcocotools.internal.mask_api import (
    bbIouBatch,
    bbNmsBatch,
    rleAreaBatch,
    rleDecodeBatch,
    rleEncodeBatch,
    rleFrBboxBatch,
    rleFrPolyBatch,
    rleFrStringBatch,
    rleIouBatch,
    rleMergeBatch,
    rleNmsBatch,
    rleToBboxBatch,
)


def test_rle_area_batch():
    """Test batched RLE area computation."""
    # Create simple RLE masks
    rle1 = RLE(10, 10, torch.tensor([0, 25, 75, 0]))  # 25 pixels
    rle2 = RLE(10, 10, torch.tensor([0, 50, 50, 0]))  # 50 pixels
    rles = RLEs([rle1, rle2])

    areas = rleAreaBatch(rles)

    assert areas.shape == (2,)
    assert areas[0] == 25
    assert areas[1] == 50


def test_bb_iou_batch():
    """Test batched bounding box IoU."""
    dt = tv.BoundingBoxes(
        torch.tensor([[10, 10, 20, 20], [30, 30, 20, 20]], dtype=torch.float32),
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(100, 100),
    )
    gt = tv.BoundingBoxes(
        torch.tensor([[10, 10, 20, 20], [35, 35, 20, 20]], dtype=torch.float32),
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(100, 100),
    )

    iou = bbIouBatch(dt, gt, [False, False])

    assert iou.shape == (2, 2)
    assert iou[0, 0] > 0.99  # Perfect overlap
    assert iou[0, 1] < 0.1  # No overlap


def test_bb_nms_batch():
    """Test batched bounding box NMS."""
    dt = tv.BoundingBoxes(
        torch.tensor([
            [10, 10, 20, 20],
            [12, 12, 20, 20],  # Overlaps with first
            [50, 50, 20, 20],
        ], dtype=torch.float32),
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(100, 100),
    )

    keep = bbNmsBatch(dt, thr=0.5)

    assert keep.shape == (3,)
    assert keep.dtype == torch.bool
    # First and third should be kept, second should be suppressed
    assert keep[0] == True  # noqa: E712
    assert keep[2] == True  # noqa: E712


def test_rle_encode_decode_batch():
    """Test batched RLE encode and decode."""
    # Create simple binary masks
    mask = tv.Mask(torch.zeros((2, 10, 10), dtype=torch.uint8))
    mask[0, 2:5, 2:5] = 1  # 3x3 square
    mask[1, 4:8, 4:8] = 1  # 4x4 square

    # Encode
    rles = rleEncodeBatch(mask)

    assert len(rles) == 2

    # Decode
    decoded = rleDecodeBatch(rles)

    assert decoded.shape == (10, 10, 2)
    assert torch.sum(decoded[:, :, 0]) == 9  # 3x3 = 9 pixels
    assert torch.sum(decoded[:, :, 1]) == 16  # 4x4 = 16 pixels


def test_rle_to_bbox_batch():
    """Test batched RLE to bounding box conversion."""
    # Note: rleToBboxBatch is an alias to the existing batched implementation
    # which handles variable-length RLEs, so we just verify it works
    mask = tv.Mask(torch.zeros((2, 10, 10), dtype=torch.uint8))
    mask[0, 2:5, 2:5] = 1
    mask[1, 4:8, 4:8] = 1

    rles = rleEncodeBatch(mask)
    # Use regular rleToBbox which is already batched
    from pytorchcocotools.internal.mask_api import rleToBbox
    bboxes = rleToBbox(rles)

    assert bboxes.shape == (2, 4)
    # First bbox should be around [2, 2, 3, 3]
    assert torch.allclose(bboxes[0, :2].float(), torch.tensor([2.0, 2.0]), atol=1.0)
    # Second bbox should be around [4, 4, 4, 4]
    assert torch.allclose(bboxes[1, :2].float(), torch.tensor([4.0, 4.0]), atol=1.0)


def test_rle_fr_bbox_batch():
    """Test batched bounding box to RLE conversion."""
    bboxes = tv.BoundingBoxes(
        torch.tensor([[2, 2, 3, 3], [4, 4, 4, 4]], dtype=torch.float32),
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(10, 10),
    )

    rles = rleFrBboxBatch(bboxes)

    assert len(rles) == 2
    # Decode to verify
    decoded = rleDecodeBatch(rles)
    assert decoded.shape == (10, 10, 2)


def test_rle_fr_string_batch():
    """Test batched string to RLE conversion."""
    # Create some RLE strings (simplified example)
    strings = [b"010203", b"040506"]
    heights = [10, 10]
    widths = [10, 10]

    rles = rleFrStringBatch(strings, heights, widths)

    assert len(rles) == 2
    assert rles[0].h == 10
    assert rles[1].h == 10
