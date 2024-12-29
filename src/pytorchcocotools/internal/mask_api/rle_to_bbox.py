import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


def rleToBbox(  # noqa: N802,
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> tv.BoundingBoxes:
    """Get bounding boxes surrounding encoded masks.

    Args:
        rles: The RLE encoded masks.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        List of bounding boxes in format [x y w h]
    """
    n = len(rles)
    device = rles[0].cnts.device if device is None else device
    bb = torch.zeros((n, 4), dtype=torch.int32, device=device)
    for i in range(n):
        if len(rles[i].cnts) == 0:  # m
            continue
        device = rles[i].cnts.device if device is None else device
        h, _w, m = rles[i].h, rles[i].w, len(rles[i].cnts)
        # m = (m // 2) * 2
        if m == 0:
            continue

        # Cumulative sum of counts for locating starts and ends
        csum = rles[i].cnts.cumsum(dim=0)

        # Identify foreground segments (odd indices and non-zero counts)
        indices = torch.arange(m, device=device)
        fg_mask = ((indices % 2) == 1) & (rles[i].cnts != 0)

        # If no foreground segments, bounding box is [0,0,0,0]
        if not fg_mask.any():
            continue

        # Gather indices for foreground segments
        fg_indices = indices[fg_mask]

        # Compute start and end positions for each foreground segment
        starts = csum[fg_indices] - rles[i].cnts[fg_indices]
        ends = csum[fg_indices] - 1

        # Convert to (x, y) coordinates
        y_starts = starts % h
        x_starts = (starts - y_starts) // h
        y_ends = ends % h
        x_ends = (ends - y_ends) // h

        # Compute overall min and max for x
        xs = x_starts.min()
        xe = x_ends.max()

        # If any foreground segment spans multiple columns, y covers the entire height
        spans_multiple_columns = (x_starts < x_ends).any()
        if spans_multiple_columns:
            ys = 0
            ye = h - 1
        else:
            # Otherwise, compute min and max for y
            ys = torch.minimum(y_starts.min(), y_ends.min())
            ye = torch.maximum(y_starts.max(), y_ends.max())

        width_ = xe - xs + 1
        height_ = ye - ys + 1

        bb[i] = torch.tensor([xs, ys, width_, height_], dtype=torch.float32, device=device)

    return tv.BoundingBoxes(bb, format=tv.BoundingBoxFormat.XYWH, canvas_size=(rles[0].h, rles[0].w))  # pyright: ignore[reportCallIssue]
