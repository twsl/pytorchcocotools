import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


def _pack_rle_counts(rles: RLEs) -> tuple[int, int, torch.device, Tensor, Tensor, Tensor] | None:
    """Pack variable-length RLE count tensors into padded tensors once per merge.

    Replaces ``pad_sequence`` with direct tensor construction to avoid the
    overhead of the ``torch.nn.utils.rnn`` helper for small batch sizes.
    """
    h, w = rles[0].h, rles[0].w
    device = rles[0].cnts.device

    count_tensors = [rles[0].cnts.to(dtype=torch.long, device=device)]
    max_len = count_tensors[0].numel()
    for i in range(1, len(rles)):
        rle = rles[i]
        if rle.h != h or rle.w != w:
            return None
        t = rle.cnts.to(dtype=torch.long, device=device)
        count_tensors.append(t)
        if t.numel() > max_len:
            max_len = t.numel()

    n = len(count_tensors)
    padded_counts = torch.zeros(n, max_len, dtype=torch.long, device=device)
    lengths = torch.empty(n, dtype=torch.long, device=device)
    for i, t in enumerate(count_tensors):
        padded_counts[i, : t.numel()] = t
        lengths[i] = t.numel()

    total_pixels = torch.tensor(h * w, dtype=torch.long, device=device)
    return h, w, device, padded_counts, lengths, total_pixels


def _rle_merge_events_buffer_eager(
    padded_counts: Tensor,
    lengths: Tensor,
    intersect: bool,
    total_pixels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Merge packed RLE counts using a vectorized event sweep.

    Foreground intervals are materialized as +1/-1 events, sorted and reduced
    into a fixed-size boundary buffer, then converted back into alternating
    background/foreground run lengths.

    Optimizations applied:
    - ``masked_fill`` replaces ``where`` + sentinel allocation (3 ops → 1).
    - ``valid.to(long)`` replaces ``where(mask, ones_like, zeros_like)``
      (3-4 ops → 1), fusing the delta construction.
    - Multiply-mask ``val * cond.to(long)`` replaces
      ``where(cond, val, zeros_like)`` for segment lengths (fused).
    - Bitwise ``>> 1`` replaces ``torch.div(…, rounding_mode="floor")``
      for integer halving.
    - ``torch.full_like`` replaces ``zeros_like + scalar`` where the fill value
      is a compile-time constant equivalent.
    """
    device = padded_counts.device
    num_masks = padded_counts.shape[0]
    sentinel_val = total_pixels + 1

    # Cumulative positions from run lengths
    cumulative = padded_counts.cumsum(dim=1)

    # Extract foreground interval boundaries
    end_pos = cumulative[:, 1::2]
    max_fg = end_pos.shape[1]
    start_pos = cumulative[:, 0::2][:, :max_fg]

    # Validity mask – single arange, fused comparison
    fg_counts = lengths >> 1
    valid = torch.arange(max_fg, device=device).unsqueeze(0) < fg_counts.unsqueeze(1)

    # Event positions: masked_fill avoids separate sentinel tensor allocation
    not_valid = ~valid
    flat_pos = torch.cat(
        [
            start_pos.masked_fill(not_valid, sentinel_val).reshape(-1),
            end_pos.masked_fill(not_valid, sentinel_val).reshape(-1),
            torch.stack([torch.zeros((), dtype=torch.long, device=device), total_pixels]),
        ],
        dim=0,
    )

    # Event deltas: cast validity mask to long (+1/0), negate for ends (-1/0)
    valid_flat = valid.to(torch.long).reshape(-1)
    flat_deltas = torch.cat(
        [valid_flat, -valid_flat, torch.zeros(2, dtype=torch.long, device=device)],
        dim=0,
    )

    # Sort events by position (stable preserves insertion order for ties)
    sort_idx = flat_pos.argsort(stable=True)
    sorted_pos = flat_pos[sort_idx]
    sorted_del = flat_deltas[sort_idx]

    # Group duplicate positions via cumsum-based ID assignment
    n = sorted_pos.shape[0]
    new_group = torch.ones(n, dtype=torch.bool, device=device)
    new_group[1:] = sorted_pos[1:] != sorted_pos[:-1]
    group_ids = new_group.to(torch.long).cumsum(0) - 1
    num_groups = group_ids[-1] + 1

    # Scatter-reduce to unique positions and summed deltas
    unique_pos = torch.zeros_like(sorted_pos) + sentinel_val
    unique_pos.scatter_reduce_(0, group_ids, sorted_pos, reduce="amin", include_self=True)
    unique_del = torch.zeros_like(sorted_del)
    unique_del.scatter_add_(0, group_ids, sorted_del)

    # Valid-group mask and cumulative active count
    valid_gp = (torch.arange(n, device=device) < num_groups) & (unique_pos <= total_pixels)
    active = unique_del.cumsum(0)

    # Segment properties – multiply-mask fuses where(cond, val, zeros) → val * cond
    both_valid = valid_gp[:-1] & valid_gp[1:]
    seg_len = (unique_pos[1:] - unique_pos[:-1]) * both_valid.to(torch.long)
    fg = (active[:-1] == num_masks) if intersect else (active[:-1] > 0)
    seg_fg = both_valid & fg

    # Pad with background state at both ends
    z_l = torch.zeros(1, dtype=torch.long, device=device)
    z_b = torch.zeros(1, dtype=torch.bool, device=device)
    segs = torch.cat([z_l, seg_len, z_l])
    states = torch.cat([z_b, seg_fg, z_b])

    # Collapse consecutive same-state runs via scatter_add
    run_start = torch.ones_like(states)
    run_start[1:] = states[1:] != states[:-1]
    run_ids = run_start.to(torch.long).cumsum(0) - 1
    run_count = run_ids[-1] + 1

    run_buf = torch.zeros_like(segs)
    run_buf.scatter_add_(0, run_ids, segs)
    return run_buf, run_count


@torch.compile(dynamic=True)
def _rle_merge_events_buffer(
    padded_counts: Tensor,
    lengths: Tensor,
    intersect: bool,
    total_pixels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compiled wrapper around the eager event-sweep implementation."""
    return _rle_merge_events_buffer_eager(padded_counts, lengths, intersect, total_pixels)


def _trim_run_buffer(run_buffer: Tensor, run_count: Tensor) -> Tensor:
    """Trim the fixed-size run buffer to the actual merged RLE counts.

    Uses Python scalars to avoid tensor-op overhead outside the compiled kernel.
    """
    n = int(run_count.item())
    if n > 1 and run_buffer[n - 1].item() == 0:
        n -= 1
    return run_buffer[:n]


@torch.inference_mode()
def rleMerge(  # noqa: N802
    rles: RLEs,
    intersect: bool,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool = False,
) -> RLE:
    """Compute union or intersection of encoded masks.

    Uses a pure-PyTorch event sweep over packed run tensors. Inputs are padded
    once, then a fixed-shape compiled kernel merges all masks.

    Args:
        rles: The masks to merge.
        intersect: Whether to compute the intersection.
        device: Kept for API compatibility; merged counts stay on the source device.
        requires_grad: Kept for API compatibility.

    Returns:
        The merged mask.
    """
    n = len(rles)
    if not rles or n == 0:
        return RLE(0, 0, torch.empty(0, dtype=torch.long))
    if n == 1:
        return rles[0]

    packed = _pack_rle_counts(rles)
    if packed is None:
        return RLE(0, 0, torch.empty(0, dtype=torch.long, device=rles[0].cnts.device))

    h, w, src_device, padded_counts, lengths, total_pixels = packed
    run_buffer, run_count = _rle_merge_events_buffer(padded_counts, lengths, intersect, total_pixels)
    counts = _trim_run_buffer(run_buffer, run_count)
    return RLE(h, w, counts.to(dtype=torch.long, device=src_device))
