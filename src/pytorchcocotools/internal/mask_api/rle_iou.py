import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs
from pytorchcocotools.internal.mask_api.bb_iou import bbIou
from pytorchcocotools.internal.mask_api.rle_area import rleArea
from pytorchcocotools.internal.mask_api.rle_to_bbox import rleToBbox


def _build_fg_intervals(rles: RLEs, src_device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Convert RLEs to padded foreground-interval tensors.

    Returns:
        starts: [N, K] – start pixel index of each foreground run (padding = -1)
        ends:   [N, K] – end pixel index (inclusive) of each foreground run (padding = -1)
        areas:  [N]    – total foreground pixel count
    """
    n = len(rles)
    all_starts: list[Tensor] = []
    all_ends: list[Tensor] = []
    all_areas: list[int] = []

    for rle in rles:
        cnts = rle.cnts.to(dtype=torch.long, device=src_device)
        if cnts.numel() == 0:
            all_starts.append(cnts.new_empty(0))
            all_ends.append(cnts.new_empty(0))
            all_areas.append(0)
            continue
        cums = torch.cumsum(cnts, dim=0)
        # Foreground runs are at odd indices (0-based); cumsum before = start, cumsum = end+1
        # bg: cnts[0], fg: cnts[1], bg: cnts[2], fg: cnts[3], ...
        # fg run i: start = cums[2*i], end = cums[2*i+1] - 1  (for i=0,1,...)
        fg_end_idx = torch.arange(1, cnts.numel(), 2, device=src_device)  # indices 1,3,5...
        fg_start_idx = fg_end_idx - 1  # indices 0,2,4...
        if fg_end_idx.numel() == 0:
            all_starts.append(cnts.new_empty(0))
            all_ends.append(cnts.new_empty(0))
            all_areas.append(0)
        else:
            fg_starts = cums[fg_start_idx]  # start of each fg run
            fg_ends = cums[fg_end_idx] - 1  # end of each fg run (inclusive)
            # Only keep non-empty runs
            non_empty = cnts[fg_end_idx] > 0
            all_starts.append(fg_starts[non_empty])
            all_ends.append(fg_ends[non_empty])
            all_areas.append(int((fg_ends[non_empty] - fg_starts[non_empty] + 1).sum().item()))

    max_k = max((s.numel() for s in all_starts), default=0)
    if max_k == 0:
        starts_t = torch.full((n, 1), -1, dtype=torch.long, device=src_device)
        ends_t = torch.full((n, 1), -1, dtype=torch.long, device=src_device)
    else:
        starts_t = torch.full((n, max_k), -1, dtype=torch.long, device=src_device)
        ends_t = torch.full((n, max_k), -1, dtype=torch.long, device=src_device)
        for i, (s, e) in enumerate(zip(all_starts, all_ends)):
            k = s.numel()
            if k > 0:
                starts_t[i, :k] = s
                ends_t[i, :k] = e

    areas_t = torch.tensor(all_areas, dtype=torch.long, device=src_device)
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
    same_size = torch.tensor(
        [[dt[d].h == gt[g].h and dt[d].w == gt[g].w for g in range(n)] for d in range(m)],
        dtype=torch.bool,
        device=src_device,
    )
    # Where bbox overlap is positive but sizes mismatch: set to -1
    size_mismatch = (o > 0) & ~same_size
    iou = torch.where(size_mismatch, torch.full_like(iou, -1.0), iou)
    # Where no bbox overlap: IoU is 0 (already 0 from interval calculation)
    iou = torch.where(o <= 0, torch.zeros_like(iou), iou)

    return iou.double()  # match original float64 output
