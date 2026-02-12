import torch
from torchvision import tv_tensors as tv
from torchvision.ops import box_convert, nms

from pytorchcocotools.internal.mask_api.bb_iou import bbIou


@torch.no_grad
def bbNms(dt: tv.BoundingBoxes, thr: float) -> list[bool]:  # noqa: N802
    """Compute non-maximum suppression between bounding boxes.

    Uses torchvision.ops.nms for vectorized NMS.

    Args:
        dt: The detected bounding boxes (shape: [n, 4]).
        thr: The IoU threshold for non-maximum suppression.

    Returns:
        A list of bools indicating which boxes to keep.
    """
    n = dt.shape[0]
    if n == 0:
        return []
    # Convert XYWH -> XYXY for torchvision nms
    xyxy = box_convert(dt.float(), in_fmt="xywh", out_fmt="xyxy")
    # NMS needs scores; use inverse index as proxy (earlier boxes have higher priority)
    scores = torch.arange(n, 0, -1, dtype=torch.float32, device=dt.device)
    keep_indices = nms(xyxy, scores, thr)
    keep = [False] * n
    for idx in keep_indices.tolist():
        keep[idx] = True
    return keep
