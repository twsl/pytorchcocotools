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


def rle_to_bbox_batch(
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> tv.BoundingBoxes:
    # """Vectorized RLE-to-bounding-box conversion using a batch dimension.

    # Args:
    #     heights (torch.Tensor): A 1D tensor of shape (B,) containing image heights for each sample.
    #     widths (torch.Tensor): A 1D tensor of shape (B,) containing image widths for each sample.
    #     counts (torch.Tensor): A 2D tensor of shape (B, M) containing run-length counts for each sample,
    #                            padded with zeros for samples whose counts are shorter than M.
    #                            The sequence for each sample should be [cnt_0, cnt_1, ..., cnt_(m-1), 0, 0, ...].

    # Returns:
    #     A tensor of shape (B, 4) containing bounding boxes [x_min, y_min, width, height] for each sample.

    # Note:
    #     1. This function assumes that each row in 'counts' alternates between background (even index)
    #        and foreground (odd index) segments, just like the original RLE approach.
    #     2. If a given sample has m=0 (no counts), or no foreground segments, the bounding box is [0, 0, 0, 0].
    #     3. All calculations are done in parallel across the batch dimension.
    # """
    counts = torch.stack([rle.cnts for rle in rles], dim=0)
    heights = torch.tensor([rle.h for rle in rles], device=counts.device)
    # widths = torch.tensor([rle.w for rle in rles], device=counts.device)
    device = counts.device
    B, M = counts.shape  # noqa: N806

    # Handle edge case where B=0 or M=0
    if B == 0 or M == 0:
        return tv.BoundingBoxes(
            torch.zeros((B, 4), dtype=torch.float32, device=device),
            format=tv.BoundingBoxFormat.XYWH,
            canvas_size=(rles[0].h, rles[0].w),
            device=device,
            requires_grad=requires_grad,
        )  # pyright:ignore[reportCallIssue]

    # A "real" length per sample can be inferred if needed, but here we assume
    # any trailing zero is just empty padding for that sample's RLE data.

    # Create a range for indices [0..M-1], broadcast to match (B, M)
    indices = torch.arange(M, device=device).unsqueeze(0).expand(B, M)

    # Compute the cumulative sum of each row
    csum = counts.cumsum(dim=1)

    # Identify foreground segments: odd indices and non-zero counts
    fg_mask = ((indices % 2) == 1) & (counts != 0)

    # If no sample has any foreground segments, return zeros
    if not fg_mask.any():
        return tv.BoundingBoxes(
            torch.zeros((B, 4), dtype=torch.float32, device=device),
            format=tv.BoundingBoxFormat.XYWH,
            canvas_size=(rles[0].h, rles[0].w),
            device=device,
            requires_grad=requires_grad,
        )  # pyright:ignore[reportCallIssue]

    # For each element in counts, the start of its segment is (csum - that_count)
    # and the end is (csum - 1). We'll compute them in parallel:
    starts = csum - counts  # shape (B, M)
    ends = csum - 1  # shape (B, M)

    # Convert to (x, y) coordinates:
    # y = index % height, x = (index - y) / height
    # We'll do this with broadcasting. We have 'heights' of shape (B,).
    # We'll expand heights to (B, 1) so it broadcasts with (B, M).
    h_expanded = heights.unsqueeze(1)  # shape (B, 1)

    # Compute y_starts, x_starts, y_ends, x_ends
    y_starts = starts % h_expanded
    x_starts = (starts - y_starts) // h_expanded
    y_ends = ends % h_expanded
    x_ends = (ends - y_ends) // h_expanded

    # Mask out positions that are not foreground
    # We'll use large sentinel values for min computations and small sentinel for max
    # so they don't affect the result when the mask is False.
    INF = torch.finfo(torch.float32).eps  # noqa: N806
    NEG_INF = -INF  # noqa: N806

    masked_x_starts = torch.where(fg_mask, x_starts.float(), torch.full_like(x_starts, INF, dtype=torch.float32))
    masked_x_ends = torch.where(fg_mask, x_ends.float(), torch.full_like(x_ends, NEG_INF, dtype=torch.float32))
    masked_y_starts = torch.where(fg_mask, y_starts.float(), torch.full_like(y_starts, INF, dtype=torch.float32))
    masked_y_ends = torch.where(fg_mask, y_ends.float(), torch.full_like(y_ends, NEG_INF, dtype=torch.float32))

    # Compute min/max for x and y per sample
    xs = masked_x_starts.min(dim=1).values
    xe = masked_x_ends.max(dim=1).values
    ys = masked_y_starts.min(dim=1).values
    ye = masked_y_ends.max(dim=1).values

    # Determine if any segment spans multiple columns within each batch.
    # That happens if x_start < x_end for any foreground index in a sample.
    spans_multiple_columns = (x_starts < x_ends) & fg_mask
    any_spans = spans_multiple_columns.any(dim=1)

    # Where a sample has a segment spanning multiple columns, set y to [0..h-1].
    # We'll use a torch.where with condition 'any_spans':
    full_height_ys = torch.zeros_like(ys)
    full_height_ye = heights - 1

    final_ys = torch.where(any_spans, full_height_ys, ys)
    final_ye = torch.where(any_spans, full_height_ye, ye)

    # For cases where a sample had no foreground segments at all, the min would be INF and max would be -INF.
    # If xs == INF or xe == -INF, we interpret it as no valid bounding box:
    no_fg = (xs == INF) | (xe == NEG_INF)
    # Set bounding box to zero in those cases:
    xs = torch.where(no_fg, torch.zeros_like(xs), xs)
    xe = torch.where(no_fg, torch.zeros_like(xe), xe)
    final_ys = torch.where(no_fg, torch.zeros_like(final_ys), final_ys)
    final_ye = torch.where(no_fg, torch.zeros_like(final_ye), final_ye)

    # Compute width and height
    width = (xe - xs + 1).clamp(min=0)  # clamp to avoid negative
    height = (final_ye - final_ys + 1).clamp(min=0)

    # Assemble final bounding boxes
    bboxes = torch.stack([xs, final_ys, width, height], dim=1)

    return tv.BoundingBoxes(
        bboxes,
        format=tv.BoundingBoxFormat.XYWH,
        canvas_size=(rles[0].h, rles[0].w),
        device=device,
        requires_grad=requires_grad,
    )  # pyright:ignore[reportCallIssue]
