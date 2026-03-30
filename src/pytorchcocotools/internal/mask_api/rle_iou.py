import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pytorchcocotools.internal.entities import RLE, RLEs
from pytorchcocotools.internal.mask_api.bb_iou import bbIou
from pytorchcocotools.internal.mask_api.rle_area import rleArea
from pytorchcocotools.internal.mask_api.rle_to_bbox import rleToBbox


def _build_fg_intervals(rles: RLEs, src_device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Convert RLEs to padded foreground-interval tensors.

    Uses batch ``cumsum`` over padded count matrix to avoid per-mask Python loops
    for interval extraction.

    Returns:
        starts: [N, K] – start pixel index of each foreground run (padding = -1)
        ends:   [N, K] – end pixel index (inclusive) of each foreground run (padding = -1)
        areas:  [N]    – total foreground pixel count
    """
    n = len(rles)
    if n == 0:
        return (
            torch.full((0, 1), -1, dtype=torch.long, device=src_device),
            torch.full((0, 1), -1, dtype=torch.long, device=src_device),
            torch.zeros(0, dtype=torch.long, device=src_device),
        )

    # Collect count tensors and pad into [n, max_len]
    count_list = [rle.cnts.to(dtype=torch.long, device=src_device) for rle in rles]
    lengths = torch.tensor([c.numel() for c in count_list], dtype=torch.long, device=src_device)
    max_len = int(lengths.max().item())

    if max_len == 0:
        return (
            torch.full((n, 1), -1, dtype=torch.long, device=src_device),
            torch.full((n, 1), -1, dtype=torch.long, device=src_device),
            torch.zeros(n, dtype=torch.long, device=src_device),
        )

    padded = pad_sequence(count_list, batch_first=True)  # [n, max_len]

    # Batch cumulative sum
    cums = torch.cumsum(padded, dim=1)  # [n, max_len]

    # Foreground runs live at odd column indices (1, 3, 5, …)
    max_fg = max_len // 2
    if max_fg == 0:
        return (
            torch.full((n, 1), -1, dtype=torch.long, device=src_device),
            torch.full((n, 1), -1, dtype=torch.long, device=src_device),
            torch.zeros(n, dtype=torch.long, device=src_device),
        )

    fg_col = torch.arange(1, max_len, 2, device=src_device)  # [max_fg]
    start_col = fg_col - 1  # even indices: 0, 2, 4, …

    fg_starts = cums[:, start_col]          # [n, max_fg]
    fg_ends = cums[:, fg_col] - 1           # [n, max_fg]
    fg_counts = padded[:, fg_col]           # [n, max_fg]

    # Valid mask: column index within this RLE's length AND run is non-empty
    valid = (fg_col.unsqueeze(0) < lengths.unsqueeze(1)) & (fg_counts > 0)  # [n, max_fg]

    starts_t = torch.where(valid, fg_starts, torch.tensor(-1, dtype=torch.long, device=src_device))
    ends_t = torch.where(valid, fg_ends, torch.tensor(-1, dtype=torch.long, device=src_device))
    areas_t = torch.where(valid, fg_ends - fg_starts + 1, torch.tensor(0, dtype=torch.long, device=src_device)).sum(dim=1)

    return starts_t, ends_t, areas_t


@torch.inference_mode()
def rleIou(dt: RLEs, gt: RLEs, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between masks.

    Vectorized interval-intersection approach:
    1. Convert each RLE to padded (start, end) foreground-interval tensors.
    2. Use broadcasting to compute ALL pairwise interval intersections simultaneously.
    3. Sum over interval dimensions to get per-pair intersection counts.
    This replaces the O(D×G×R) nested Python loop with O(D×G×K×L) tensor ops.

    Args:
        dt: The RLE encoded detection masks.
        gt: The RLE encoded ground truth masks.
        iscrowd: The crowd label for each ground truth mask.

    Returns:
        The intersection over union between the masks. Shape [M, N].
    """
    m = len(dt)
    n = len(gt)
    if m == 0 or n == 0:
        return torch.zeros((m, n), dtype=torch.float64)

    src_device = dt[0].cnts.device

    # Bounding-box pre-filter: zero out pairs with no bbox overlap (fast vectorised path)
    db = rleToBbox(dt)
    gb = rleToBbox(gt)
    o = bbIou(db, gb, iscrowd)  # [m, n]; 0 where no bbox overlap

    # Build foreground-interval tensors for all dt and gt masks
    dt_starts, dt_ends, dt_areas = _build_fg_intervals(dt, src_device)  # [m, K]
    gt_starts, gt_ends, gt_areas = _build_fg_intervals(gt, src_device)  # [n, L]

    K = dt_starts.shape[1]
    L = gt_starts.shape[1]

    # Broadcast to [m, n, K, L] for all-pairs interval intersection
    # dt: [m, 1, K, 1],  gt: [1, n, 1, L]
    ds = dt_starts[:, None, :, None]  # [m, 1, K, 1]
    de = dt_ends[:, None, :, None]  # [m, 1, K, 1]
    gs = gt_starts[None, :, None, :]  # [1, n, 1, L]
    ge = gt_ends[None, :, None, :]  # [1, n, 1, L]

    # Overlap (in pixels) between each dt-interval and each gt-interval
    overlap = (torch.minimum(de, ge) - torch.maximum(ds, gs) + 1).clamp(min=0)  # [m, n, K, L]

    # Zero out padded entries (padding sentinel = -1)
    valid = (ds >= 0) & (gs >= 0)  # [m, n, K, L]
    # Use float64 (double) to match reference implementation's integer-arithmetic precision
    intersection = (overlap * valid).sum(dim=(2, 3)).double()  # [m, n]

    # Union: dt_area + gt_area - intersection  (for normal GTs)
    # For crowd GTs: union = dt_area only
    iscrowd_t = torch.tensor(iscrowd, dtype=torch.bool, device=src_device)
    dt_area_f = dt_areas.double()  # [m]
    gt_area_f = gt_areas.double()  # [n]

    union = dt_area_f[:, None] + gt_area_f[None, :] - intersection  # [m, n]
    # For crowd GTs, override union with dt area
    union = torch.where(iscrowd_t[None, :], dt_area_f[:, None].expand(m, n), union)

    # IoU: 0 when both are empty (avoid division by zero)
    iou = torch.where(
        (intersection == 0) & (union == 0),
        torch.zeros_like(union),
        intersection / union.clamp(min=1),
    )

    # Apply bbox pre-filter mask and dimension/size guard from original
    # For mismatched h/w: set to -1 (matching original behaviour)
    dt_hw = torch.tensor([(r.h, r.w) for r in dt], dtype=torch.long, device=src_device)  # [m, 2]
    gt_hw = torch.tensor([(r.h, r.w) for r in gt], dtype=torch.long, device=src_device)  # [n, 2]
    same_size = (dt_hw[:, None, :] == gt_hw[None, :, :]).all(dim=2)  # [m, n]

    # Where bbox overlap is positive but sizes mismatch: set to -1
    size_mismatch = (o > 0) & ~same_size
    iou = torch.where(size_mismatch, torch.full_like(iou, -1.0), iou)
    # Where no bbox overlap: IoU is 0 (already 0 from interval calculation)
    iou = torch.where(o <= 0, torch.zeros_like(iou), iou)

    return iou.double()  # match original float64 output
