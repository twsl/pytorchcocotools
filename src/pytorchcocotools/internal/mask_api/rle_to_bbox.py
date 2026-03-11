import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.no_grad
def rleToBbox(  # noqa: N802,
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> tv.BoundingBoxes:
    """Get bounding boxes surrounding encoded masks.

    Batched: pads all RLE count vectors to the same length, then computes
    all bounding boxes in a single set of vectorised tensor operations.

    Args:
        rles: The RLE encoded masks.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        List of bounding boxes in format [x y w h]
    """
    n = len(rles)
    canvas_h = rles[0].h
    canvas_w = rles[0].w
    device = rles[0].cnts.device if device is None else device

    lengths = [len(r.cnts) for r in rles]
    max_len = max(lengths) if lengths else 0

    zeros_n4 = tv.BoundingBoxes(
        torch.zeros((n, 4), dtype=torch.float32, device=device),
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(canvas_h, canvas_w),
    )  # pyright: ignore[reportCallIssue]

    if max_len == 0:
        return zeros_n4

    # Pad counts to max_len with zeros (zero-count segments are silently ignored
    # by the fg_mask below, so padding is safe).
    counts = torch.zeros((n, max_len), dtype=torch.long, device=device)
    for i, r in enumerate(rles):
        ml = lengths[i]
        if ml > 0:
            counts[i, :ml] = r.cnts.to(device=device, dtype=torch.long)

    heights = torch.tensor([r.h for r in rles], device=device, dtype=torch.long)

    B, M = counts.shape  # noqa: N806

    # Cumulative sum per row.
    csum = counts.cumsum(dim=1)  # [B, M]

    # Foreground segments: odd column index AND non-zero count.
    indices = torch.arange(M, device=device).unsqueeze(0).expand(B, M)  # [B, M]
    fg_mask = ((indices % 2) == 1) & (counts != 0)  # [B, M]

    if not fg_mask.any():
        return zeros_n4

    # Pixel-level start / end for each segment.
    starts = csum - counts  # [B, M]
    ends = csum - 1  # [B, M]

    h_exp = heights.unsqueeze(1)  # [B, 1]
    y_starts = starts % h_exp
    x_starts = (starts - y_starts) // h_exp
    y_ends = ends % h_exp
    x_ends = (ends - y_ends) // h_exp

    # Use ±inf sentinels so masked-out slots don't affect min/max.
    POS_INF = float("inf")  # noqa: N806
    NEG_INF = float("-inf")  # noqa: N806

    mx_s = x_starts.float().masked_fill(~fg_mask, POS_INF)
    mx_e = x_ends.float().masked_fill(~fg_mask, NEG_INF)
    my_s = y_starts.float().masked_fill(~fg_mask, POS_INF)
    my_e = y_ends.float().masked_fill(~fg_mask, NEG_INF)

    xs = mx_s.min(dim=1).values  # [B]
    xe = mx_e.max(dim=1).values  # [B]
    ys = my_s.min(dim=1).values  # [B]
    ye = my_e.max(dim=1).values  # [B]

    # If any fg segment spans multiple columns the y-extent is the full height.
    spans_cols = ((x_starts < x_ends) & fg_mask).any(dim=1)  # [B]
    heights_f = heights.float()
    ys = torch.where(spans_cols, torch.zeros_like(ys), ys)
    ye = torch.where(spans_cols, heights_f - 1.0, ye)

    # Rows with no foreground at all → zero bbox.
    no_fg = ~fg_mask.any(dim=1)  # [B]
    xs = torch.where(no_fg, torch.zeros_like(xs), xs)
    xe = torch.where(no_fg, torch.zeros_like(xe), xe)
    ys = torch.where(no_fg, torch.zeros_like(ys), ys)
    ye = torch.where(no_fg, torch.zeros_like(ye), ye)

    width = (xe - xs + 1).clamp(min=0)
    height = (ye - ys + 1).clamp(min=0)

    # Match the original int32 dtype so downstream users and tests stay consistent.
    bboxes = torch.stack([xs, ys, width, height], dim=1).to(torch.int32)  # [B, 4]
    return tv.BoundingBoxes(
        bboxes,
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(canvas_h, canvas_w),
        device=device,
        requires_grad=requires_grad,
    )  # pyright: ignore[reportCallIssue]
