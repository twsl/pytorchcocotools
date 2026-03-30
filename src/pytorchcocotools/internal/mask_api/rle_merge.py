import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


def _pack_rle_counts(rles: RLEs) -> tuple[int, int, torch.device, Tensor, Tensor, Tensor] | None:
    """Pack variable-length RLE count tensors into padded tensors once per merge."""
    h, w = rles[0].h, rles[0].w
    src_device = rles[0].cnts.device

    count_tensors = [rles[0].cnts.to(dtype=torch.long, device=src_device)]
    for i in range(1, len(rles)):
        rle = rles[i]
        if rle.h != h or rle.w != w:
            return None
        count_tensors.append(rle.cnts.to(dtype=torch.long, device=src_device))

    lengths = torch.tensor([counts.numel() for counts in count_tensors], dtype=torch.long, device=src_device)
    padded_counts = pad_sequence(count_tensors, batch_first=True)
    total_pixels = torch.tensor(h * w, dtype=torch.long, device=src_device)
    return h, w, src_device, padded_counts, lengths, total_pixels


def _rle_merge_events_buffer_eager(
    padded_counts: Tensor,
    lengths: Tensor,
    intersect: bool,
    total_pixels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Merge packed RLE counts using a fixed-shape event sweep.

    Foreground intervals are materialized as +1/-1 events, reduced into a
    fixed-size boundary buffer, and then converted back into alternating
    background/foreground run lengths. The buffer shape depends only on the
    padded input shape, which keeps the kernel compatible with `torch.compile`.
    """
    device = padded_counts.device
    num_masks = padded_counts.shape[0]

    cumulative_counts = torch.cumsum(padded_counts, dim=1)
    end_positions = cumulative_counts[:, 1::2]
    foreground_counts = torch.div(lengths, 2, rounding_mode="floor")
    max_foreground_runs = end_positions.shape[1]
    start_positions = cumulative_counts[:, 0::2][:, :max_foreground_runs]
    interval_mask = torch.arange(max_foreground_runs, device=device).unsqueeze(0) < foreground_counts.unsqueeze(1)

    sentinel = torch.zeros((num_masks, max_foreground_runs), dtype=torch.long, device=device) + (total_pixels + 1)
    flat_positions = torch.cat(
        [
            torch.where(interval_mask, start_positions, sentinel).reshape(-1),
            torch.where(interval_mask, end_positions, sentinel).reshape(-1),
            torch.stack((torch.zeros((), dtype=torch.long, device=device), total_pixels)),
        ],
        dim=0,
    )
    flat_deltas = torch.cat(
        [
            torch.where(
                interval_mask,
                torch.ones_like(start_positions, dtype=torch.long),
                torch.zeros_like(start_positions, dtype=torch.long),
            ).reshape(-1),
            torch.where(
                interval_mask,
                -torch.ones_like(end_positions, dtype=torch.long),
                torch.zeros_like(end_positions, dtype=torch.long),
            ).reshape(-1),
            torch.zeros(2, dtype=torch.long, device=device),
        ],
        dim=0,
    )

    sort_order = torch.argsort(flat_positions, stable=True)
    sorted_positions = flat_positions[sort_order]
    sorted_deltas = flat_deltas[sort_order]

    new_group = torch.ones_like(sorted_positions, dtype=torch.bool)
    new_group[1:] = sorted_positions[1:] != sorted_positions[:-1]
    group_ids = torch.cumsum(new_group.to(torch.long), dim=0) - 1
    group_count = group_ids[-1] + 1

    unique_positions = torch.zeros_like(sorted_positions) + (total_pixels + 1)
    unique_positions.scatter_reduce_(0, group_ids, sorted_positions, reduce="amin", include_self=True)
    unique_deltas = torch.zeros_like(sorted_deltas)
    unique_deltas.scatter_add_(0, group_ids, sorted_deltas)

    group_mask = torch.arange(sorted_positions.shape[0], device=device) < group_count
    valid_groups = torch.logical_and(group_mask, unique_positions <= total_pixels)

    active_counts = torch.cumsum(unique_deltas, dim=0)
    valid_segments = torch.logical_and(valid_groups[:-1], valid_groups[1:])
    segment_lengths = torch.where(
        valid_segments,
        unique_positions[1:] - unique_positions[:-1],
        torch.zeros_like(unique_positions[:-1]),
    )
    active_segments = (active_counts[:-1] == num_masks) if intersect else (active_counts[:-1] > 0)
    segment_states = torch.logical_and(valid_segments, active_segments)

    zeros_long = torch.zeros(1, dtype=torch.long, device=device)
    lengths_full = torch.cat(
        [
            zeros_long,
            segment_lengths,
            zeros_long,
        ]
    )
    zeros_bool = torch.zeros(1, dtype=torch.bool, device=device)
    states_full = torch.cat(
        [
            zeros_bool,
            segment_states,
            zeros_bool,
        ]
    )

    run_start = torch.ones_like(states_full, dtype=torch.bool)
    run_start[1:] = states_full[1:] != states_full[:-1]
    run_ids = torch.cumsum(run_start.to(torch.long), dim=0) - 1
    run_count = run_ids[-1] + 1

    run_buffer = torch.zeros_like(lengths_full)
    run_buffer.scatter_add_(0, run_ids, lengths_full)
    return run_buffer, run_count


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
    """Trim the fixed-size run buffer to the actual merged RLE counts."""
    last_index = torch.clamp(run_count - 1, min=0)
    last_value = torch.gather(run_buffer, 0, last_index.unsqueeze(0)).squeeze(0)
    keep_count = run_count - torch.logical_and(run_count > 1, last_value == 0).to(torch.long)
    keep_mask = torch.arange(run_buffer.shape[0], device=run_buffer.device) < keep_count
    return run_buffer[keep_mask]


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
