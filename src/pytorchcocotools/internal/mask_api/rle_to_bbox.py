import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.compile(dynamic=True, mode="reduce-overhead")
def _compute_bboxes(counts: Tensor, heights: Tensor) -> Tensor:
    """Compute XYWH bounding boxes from padded RLE count matrix. Fully compilable kernel.

    Args:
        counts: [B, M] padded RLE counts.
        heights: [B] heights per mask.

    Returns:
        [B, 4] int32 bboxes in XYWH format.
    """
    B, M = counts.shape  # noqa: N806

    csum = counts.cumsum(dim=1)  # [B, M]

    indices = torch.arange(M, device=counts.device).unsqueeze(0).expand(B, M)  # [B, M]
    fg_mask = ((indices % 2) == 1) & (counts != 0)  # [B, M]

    starts = csum - counts  # [B, M]
    ends = csum - 1  # [B, M]

    h_exp = heights.unsqueeze(1)  # [B, 1]
    y_starts = starts % h_exp
    x_starts = (starts - y_starts) // h_exp
    y_ends = ends % h_exp
    x_ends = (ends - y_ends) // h_exp

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

    spans_cols = ((x_starts < x_ends) & fg_mask).any(dim=1)  # [B]
    heights_f = heights.float()
    ys = torch.where(spans_cols, torch.zeros_like(ys), ys)
    ye = torch.where(spans_cols, heights_f - 1.0, ye)

    no_fg = ~fg_mask.any(dim=1)  # [B]
    xs = torch.where(no_fg, torch.zeros_like(xs), xs)
    xe = torch.where(no_fg, torch.zeros_like(xe), xe)
    ys = torch.where(no_fg, torch.zeros_like(ys), ys)
    ye = torch.where(no_fg, torch.zeros_like(ye), ye)

    width = (xe - xs + 1).clamp(min=0)
    height = (ye - ys + 1).clamp(min=0)

    return torch.stack([xs, ys, width, height], dim=1).to(torch.int32)  # [B, 4]


@torch.inference_mode()
def rleToBbox(  # noqa: N802,
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool = False,
) -> tv.BoundingBoxes:
    """Get bounding boxes surrounding encoded masks.

    Batched: pads all RLE count vectors to the same length, then computes
    all bounding boxes via a torch.compiled kernel.

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

    if max_len == 0:
        return tv.BoundingBoxes(
            torch.zeros((n, 4), dtype=torch.float32, device=device),
            format=tv.BoundingBoxFormat.XYWH,
            canvas_size=(canvas_h, canvas_w),
        )  # ty:ignore[no-matching-overload]

    # Pad counts to max_len with zeros
    counts = torch.zeros((n, max_len), dtype=torch.long, device=device)
    for i, r in enumerate(rles):
        ml = lengths[i]
        if ml > 0:
            counts[i, :ml] = r.cnts.to(device=device, dtype=torch.long)

    heights = torch.tensor([r.h for r in rles], device=device, dtype=torch.long)

    bboxes = _compute_bboxes(counts, heights)

    return tv.BoundingBoxes(
        bboxes,
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(canvas_h, canvas_w),
        device=device,
        requires_grad=requires_grad,
    )  # ty:ignore[no-matching-overload]
