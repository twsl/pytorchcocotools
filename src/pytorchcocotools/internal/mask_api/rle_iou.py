import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pytorchcocotools.internal.entities import RLE, RLEs
from pytorchcocotools.internal.mask_api.bb_iou import bbIou
from pytorchcocotools.internal.mask_api.rle_area import rleArea
from pytorchcocotools.internal.mask_api.rle_to_bbox import rleToBbox
from pytorchcocotools.utils.device import get_float_dtype


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

    return _build_fg_intervals_core(padded, lengths, n, max_len, src_device)


def _build_fg_intervals_core(
    padded: Tensor, lengths: Tensor, n: int, max_len: int, src_device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Core interval extraction from padded count matrix."""
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

    fg_starts = cums[:, start_col]  # [n, max_fg]
    fg_ends = cums[:, fg_col] - 1  # [n, max_fg]
    fg_counts = padded[:, fg_col]  # [n, max_fg]

    # Valid mask: column index within this RLE's length AND run is non-empty
    valid = (fg_col.unsqueeze(0) < lengths.unsqueeze(1)) & (fg_counts > 0)  # [n, max_fg]

    minus_one = torch.tensor(-1, dtype=torch.long, device=src_device)
    starts_t = torch.where(valid, fg_starts, minus_one)
    ends_t = torch.where(valid, fg_ends, minus_one)
    areas_t = torch.where(valid, fg_ends - fg_starts + 1, torch.tensor(0, dtype=torch.long, device=src_device)).sum(
        dim=1
    )

    return starts_t, ends_t, areas_t


@torch.compile(dynamic=True)
def _rle_iou_core(
    dt_starts: Tensor,
    dt_ends: Tensor,
    dt_areas: Tensor,
    gt_starts: Tensor,
    gt_ends: Tensor,
    gt_areas: Tensor,
    iscrowd_t: Tensor,
) -> Tensor:
    """Compiled kernel for all-pairs RLE IoU via interval intersection.

    Fuses the 4D broadcast, intersection sum, union, and IoU division.
    """
    m = dt_starts.shape[0]
    n = gt_starts.shape[0]

    # Broadcast to [m, n, K, L] for all-pairs interval intersection
    ds = dt_starts[:, None, :, None]  # [m, 1, K, 1]
    de = dt_ends[:, None, :, None]  # [m, 1, K, 1]
    gs = gt_starts[None, :, None, :]  # [1, n, 1, L]
    ge = gt_ends[None, :, None, :]  # [1, n, 1, L]

    # Overlap (in pixels) between each dt-interval and each gt-interval
    overlap = (torch.minimum(de, ge) - torch.maximum(ds, gs) + 1).clamp(min=0)  # [m, n, K, L]

    # Zero out padded entries (padding sentinel = -1)
    valid = (ds >= 0) & (gs >= 0)  # [m, n, K, L]

    intersection = (overlap * valid).sum(dim=(2, 3)).to(dtype=dt_areas.dtype)  # [m, n]

    dt_area_f = dt_areas.to(dtype=dt_areas.dtype)
    gt_area_f = gt_areas.to(dtype=dt_areas.dtype)

    union = dt_area_f[:, None] + gt_area_f[None, :] - intersection
    union = torch.where(iscrowd_t[None, :], dt_area_f[:, None].expand(m, n), union)

    iou = torch.where(
        (intersection == 0) & (union == 0),
        torch.zeros_like(union),
        intersection / union.clamp(min=1),
    )
    return iou


@torch.inference_mode()
def rleIou(dt: RLEs, gt: RLEs, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between masks.

    Vectorized interval-intersection approach with compiled IoU kernel:
    1. Convert each RLE to padded (start, end) foreground-interval tensors.
    2. Use a torch.compiled kernel for fused 4D broadcast intersection + IoU.
    3. Apply bbox pre-filter and size-mismatch guards.

    Args:
        dt: The RLE encoded detection masks.
        gt: The RLE encoded ground truth masks.
        iscrowd: The crowd label for each ground truth mask.

    Returns:
        The intersection over union between the masks. Shape [M, N].
    """
    m = len(dt)
    n = len(gt)
    src_device = dt[0].cnts.device if dt else gt[0].cnts.device if gt else torch.device("cpu")
    float_dtype = get_float_dtype(src_device)
    if m == 0 or n == 0:
        return torch.zeros((m, n), dtype=float_dtype, device=src_device)

    dt_h, dt_w = dt[0].h, dt[0].w
    same_canvas = all(r.h == dt_h and r.w == dt_w for r in dt) and all(r.h == dt_h and r.w == dt_w for r in gt)

    if not same_canvas:
        db = rleToBbox(dt)
        gb = rleToBbox(gt)
        o = bbIou(db, gb, iscrowd)

    # Build foreground-interval tensors for all dt and gt masks
    dt_starts, dt_ends, dt_areas = _build_fg_intervals(dt, src_device)  # [m, K]
    gt_starts, gt_ends, gt_areas = _build_fg_intervals(gt, src_device)  # [n, L]

    iscrowd_t = torch.tensor(iscrowd, dtype=torch.bool, device=src_device)

    iou = _rle_iou_core(
        dt_starts,
        dt_ends,
        dt_areas.to(dtype=float_dtype),
        gt_starts,
        gt_ends,
        gt_areas.to(dtype=float_dtype),
        iscrowd_t,
    )

    if same_canvas:
        return iou

    # Apply bbox pre-filter mask and dimension/size guard from original
    dt_hw = torch.tensor([(r.h, r.w) for r in dt], dtype=torch.long, device=src_device)  # [m, 2]
    gt_hw = torch.tensor([(r.h, r.w) for r in gt], dtype=torch.long, device=src_device)  # [n, 2]
    same_size = (dt_hw[:, None, :] == gt_hw[None, :, :]).all(dim=2)  # [m, n]

    size_mismatch = (o > 0) & ~same_size
    iou = torch.where(size_mismatch, torch.full_like(iou, -1.0), iou)
    iou = torch.where(o <= 0, torch.zeros_like(iou), iou)

    return iou
