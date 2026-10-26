"""Batched ("ragged") RLE operations.

Many RLEs are stored as one flat ``counts`` tensor plus a ``lengths`` tensor (CSR layout), so that
string coding, polygon rasterisation and unions run as a handful of vectorised ops instead of
per-mask Python loops.
"""

import ctypes

import torch
from torch import Tensor

_MAX_CHARS_PER_COUNT = 8  # 40 bits of signed delta, enough for any h * w < 2**39


def segment_ids(lengths: Tensor) -> Tensor:
    """Return the segment index of every element of a ragged tensor with the given ``lengths``."""
    return torch.repeat_interleave(torch.arange(lengths.numel(), device=lengths.device), lengths)


def segment_offsets(lengths: Tensor) -> Tensor:
    """Return the start offset of every segment of a ragged tensor with the given ``lengths``."""
    return torch.cumsum(lengths, 0) - lengths


def ragged_take(values: Tensor, lengths: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
    """Select (and reorder) rows of a ragged tensor.

    Args:
        values: The flat values.
        lengths: The number of values per row.
        rows: The rows to take.

    Returns:
        The flat values and lengths of the selected rows.
    """
    new_lengths = lengths[rows]
    seg = segment_ids(new_lengths)
    local = torch.arange(seg.numel(), device=values.device) - segment_offsets(new_lengths)[seg]
    return values[segment_offsets(lengths)[rows][seg] + local], new_lengths


def strings_to_counts(strings: list[bytes], device: torch.device | str | None = None) -> tuple[Tensor, Tensor]:
    """Decode compressed COCO RLE strings (LEB128-like, 6 bits/char) into ragged counts.

    Args:
        strings: The compressed RLE strings.
        device: The device to decode on.

    Returns:
        The flat int64 counts and the number of counts per string.
    """
    num_strings = len(strings)
    joined = b"".join(strings)
    if not joined:
        return torch.zeros(0, dtype=torch.long, device=device), torch.zeros(
            num_strings, dtype=torch.long, device=device
        )
    chars = torch.frombuffer(bytearray(joined), dtype=torch.uint8).to(device=device, dtype=torch.long) - 48
    str_lengths = torch.tensor([len(s) for s in strings], dtype=torch.long, device=device)
    arange = torch.arange(chars.numel(), device=chars.device)

    ends = (chars & 0x20) == 0
    end_idx = ends.nonzero().squeeze(1)
    num_values = end_idx.numel()
    value_id = torch.cumsum(ends.long(), 0) - ends.long()
    value_start = torch.cat([end_idx.new_zeros(1), end_idx[:-1] + 1])
    shift = 5 * (arange - value_start[value_id])
    x = chars.new_zeros(num_values).index_add_(0, value_id, (chars & 0x1F) << shift)
    # sign extension on the last char of each value
    last_shift = shift[end_idx] + 5
    x = torch.where((chars[end_idx] & 0x10) != 0, x - (torch.ones_like(x) << last_shift), x)

    string_of_value = segment_ids(str_lengths)[end_idx]
    lengths = torch.bincount(string_of_value, minlength=num_strings)
    m = arange[:num_values] - segment_offsets(lengths)[string_of_value]
    # counts[m] = x[m] + counts[m - 2] for m > 2: two interleaved prefix sums per string
    odd = (m & 1) == 1
    even = ((m & 1) == 0) & (m >= 2)
    result = x.clone()
    for chain in (odd, even):
        vals = torch.where(chain, x, 0)
        csum = torch.cumsum(vals, 0)
        base = (csum - vals)[segment_offsets(lengths).clamp(max=max(num_values - 1, 0))]
        result = torch.where(chain, csum - base[string_of_value], result)
    return result, lengths


def counts_to_strings(counts: Tensor, lengths: Tensor) -> list[bytes]:
    """Encode ragged RLE counts into compressed COCO RLE strings.

    Args:
        counts: The flat int64 counts.
        lengths: The number of counts per RLE.

    Returns:
        One compressed string per RLE.
    """
    counts = counts.long()
    lengths = lengths.to(device=counts.device, dtype=torch.long)
    if counts.numel() == 0:
        return [b""] * lengths.numel()
    seg = segment_ids(lengths)
    idx = torch.arange(counts.numel(), device=counts.device)
    m = idx - segment_offsets(lengths)[seg]
    x = torch.where(m > 2, counts - counts[(idx - 2).clamp(min=0)], counts)

    shifts = 5 * torch.arange(_MAX_CHARS_PER_COUNT, device=counts.device)
    chunk = (x[:, None] >> shifts) & 0x1F
    rest = x[:, None] >> (shifts + 5)
    more = torch.where((chunk & 0x10) != 0, rest != -1, rest != 0)
    active = torch.cat([torch.ones_like(more[:, :1]), torch.cumprod(more[:, :-1].long(), 1).bool()], 1)
    chars = (chunk | (more.long() << 5)) + 48

    encoded = chars[active].to(torch.uint8).cpu().contiguous()
    # raw copy of the CPU buffer; avoids a slow per-element ``tolist``
    flat = ctypes.string_at(encoded.data_ptr(), encoded.numel())
    chars_per_rle = lengths.new_zeros(lengths.numel()).index_add_(0, seg, active.sum(1))
    result: list[bytes] = []
    start = 0
    for n in chars_per_rle.tolist():
        result.append(flat[start : start + n])
        start += n
    return result


def polygons_to_counts(xy: Tensor, num_vertices: Tensor, h: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """Rasterise many polygons into ragged RLE counts, matching ``rleFrPoly`` of pycocotools.

    Args:
        xy: The vertices of all polygons, shape ``[V, 2]``; their dtype sets the interpolation precision.
        num_vertices: The number of vertices per polygon.
        h: The canvas height per polygon.
        w: The canvas width per polygon.

    Returns:
        The flat int64 counts and the number of counts per polygon.
    """
    device = xy.device
    num_polys = num_vertices.numel()
    poly_of_vertex = segment_ids(num_vertices)
    # (int)(5 * x + .5) in C truncates towards zero, as does the int64 cast
    xi = (5.0 * xy[:, 0] + 0.5).long()
    yi = (5.0 * xy[:, 1] + 0.5).long()
    nxt = torch.arange(xy.shape[0], device=device) + 1
    first = segment_offsets(num_vertices)
    nxt[(first + num_vertices - 1)[num_vertices > 0]] = first[num_vertices > 0]

    xs, xe, ys, ye = xi, xi[nxt], yi, yi[nxt]
    dx = (xe - xs).abs()
    dy = (ys - ye).abs()
    x_major = dx >= dy
    flip = (x_major & (xs > xe)) | (~x_major & (ys > ye))
    xs, xe = torch.where(flip, xe, xs), torch.where(flip, xs, xe)
    ys, ye = torch.where(flip, ye, ys), torch.where(flip, ys, ye)
    dxy = torch.where(x_major, dx, dy)
    slope = torch.where(x_major, ye - ys, xe - xs).to(xy.dtype) / dxy.clamp(min=1).to(xy.dtype)

    # The reference walks every point of the 5x up-sampled boundary but only keeps steps that cross a
    # column boundary c|c+1 with c % 5 == 2. Those steps are enumerated directly instead.
    def interp(base: Tensor, s: Tensor, t: Tensor) -> Tensor:
        return (base + s * t + 0.5).long()

    # x-major edges step one column per point: crossings at t with (xs + t) % 5 == 2, t < dx
    t0 = (2 - xs) % 5
    n_x = torch.where(x_major & (t0 <= dx - 1), (dx - 1 - t0).div(5, rounding_mode="floor") + 1, 0)
    ex = segment_ids(n_x)
    tx = t0[ex] + 5 * (torch.arange(ex.numel(), device=device) - segment_offsets(n_x)[ex])
    xd_x = xs[ex] + tx
    yv_x = torch.minimum(interp(ys[ex], slope[ex], tx), interp(ys[ex], slope[ex], tx + 1))

    # y-major edges change column at most once per step: solve for the step crossing each column
    lo, hi = torch.minimum(xs, xe), torch.maximum(xs, xe)
    c0 = lo + (2 - lo) % 5
    n_y = torch.where(~x_major & (c0 <= hi - 1), (hi - 1 - c0).div(5, rounding_mode="floor") + 1, 0)
    ey = segment_ids(n_y)
    col = c0[ey] + 5 * (torch.arange(ey.numel(), device=device) - segment_offsets(n_y)[ey])
    xs_y, s_y, last = xs[ey], slope[ey], dy[ey] - 1
    rising = s_y > 0
    bound = (col + 0.5 - xs_y) / s_y
    t_est = torch.where(rising, torch.ceil(bound) - 1, torch.floor(bound)).long()
    t_lo = torch.minimum(t_est.clamp(min=0), last)
    for k in (-1, 1):
        u_a, u_b = interp(xs_y, s_y, t_lo), interp(xs_y, s_y, t_lo + 1)
        crosses = torch.where(rising, (u_a <= col) & (u_b > col), (u_a > col) & (u_b <= col))
        t_lo = torch.where(crosses, t_lo, torch.minimum((t_est + k).clamp(min=0), last))
    xd_y = col
    yv_y = ys[ey] + t_lo

    poly_c = poly_of_vertex[torch.cat([ex, ey])]
    xd = torch.cat([xd_x, xd_y])
    keep = (xd >= 2) & ((xd - 2) // 5 <= w[poly_c] - 1)
    poly_b = poly_c[keep]
    x_px = (xd[keep] - 2) // 5
    y_px = torch.cat([yv_x, yv_y])[keep].add(2).div(5, rounding_mode="floor")
    y_px = torch.minimum(y_px.clamp(min=0), h[poly_b])
    hw = h * w

    positions = torch.cat([x_px * h[poly_b] + y_px, hw])
    owners = torch.cat([poly_b, torch.arange(num_polys, device=device)])
    big = int(hw.max()) + 1 if num_polys else 1
    keys, multiplicity = torch.unique_consecutive(torch.sort(owners * big + positions).values, return_counts=True)
    owners = keys // big
    positions = keys % big
    # boundaries hit an even number of times cancel out; the h*w sentinel is always kept
    kept = ((multiplicity & 1) == 1) | (positions == hw[owners])
    return _diff_within(positions[kept], owners[kept], num_polys)


def _diff_within(positions: Tensor, owners: Tensor, num_segments: int) -> tuple[Tensor, Tensor]:
    lengths = torch.bincount(owners, minlength=num_segments)
    prev = torch.cat([positions.new_zeros(1), positions[:-1]])
    prev[segment_offsets(lengths)[lengths > 0]] = 0
    return positions - prev, lengths


def run_bounds(counts: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return start positions, foreground flags and owning RLE of every run of ragged counts."""
    seg = segment_ids(lengths)
    csum = torch.cumsum(counts, 0)
    seg_base = (csum - counts)[segment_offsets(lengths).clamp(max=max(counts.numel() - 1, 0))]
    ends = csum - seg_base[seg]
    local = torch.arange(counts.numel(), device=counts.device) - segment_offsets(lengths)[seg]
    return ends - counts, (local & 1) == 1, seg


def union_counts(counts: Tensor, lengths: Tensor, group: Tensor, hw: Tensor) -> tuple[Tensor, Tensor]:
    """Union RLEs that share a group id, matching ``rleMerge(intersect=False)`` of pycocotools.

    Args:
        counts: The flat int64 counts of all RLEs.
        lengths: The number of counts per RLE.
        group: The output group of every RLE.
        hw: The canvas size ``h * w`` per group.

    Returns:
        The flat int64 counts and the number of counts per group.
    """
    num_groups = hw.numel()
    starts, fg, owner = run_bounds(counts, lengths)
    fg &= counts > 0
    owner_group = group[owner[fg]]
    big = int(hw.max()) + 2
    keys = torch.cat([owner_group * big + starts[fg], owner_group * big + starts[fg] + counts[fg]])
    deltas = torch.cat([torch.ones_like(owner_group), -torch.ones_like(owner_group)])
    keys, order = torch.sort(keys)
    keys, inverse = torch.unique_consecutive(keys, return_inverse=True)
    deltas = torch.zeros_like(keys).index_add_(0, inverse, deltas[order])
    # every group sums to zero, so a global prefix sum is already segmented
    state = torch.cumsum(deltas, 0) > 0
    owners = keys // big
    prev_state = torch.cat([state.new_zeros(1), state[:-1]])
    prev_state[torch.cat([owners.new_ones(1).bool(), owners[1:] != owners[:-1]])] = False
    flip = state != prev_state

    # the h*w sentinel sorts after a boundary at h*w and is dropped when the last run is foreground
    bound_owner = torch.cat([owners[flip], torch.arange(num_groups, device=hw.device)])
    bound_pos = torch.cat([(keys % big)[flip], hw])
    sentinel = torch.cat([flip[flip] & False, flip.new_ones(num_groups)])
    order = torch.argsort((bound_owner * big + bound_pos) * 2 + sentinel.long())
    bound_owner, bound_pos, sentinel = bound_owner[order], bound_pos[order], sentinel[order]
    diffs, _ = _diff_within(bound_pos, bound_owner, num_groups)
    keep = ~sentinel | (diffs > 0)
    return diffs[keep], torch.bincount(bound_owner[keep], minlength=num_groups)


def rle_bboxes(counts: Tensor, lengths: Tensor, h: Tensor) -> Tensor:
    """Return inclusive ``[x0, y0, x1, y1]`` pixel boxes of ragged RLEs (empty masks give ``x1 < x0``)."""
    starts, fg, owner = run_bounds(counts, lengths)
    fg &= counts > 0
    n = lengths.numel()
    s = starts[fg]
    e = s + counts[fg] - 1
    o = owner[fg]
    ho = h[o]
    xs, xe = s // ho, e // ho
    spans = xs != xe
    ys = torch.where(spans, 0, s % ho)
    ye = torch.where(spans, ho - 1, e % ho)
    big = torch.iinfo(torch.long).max
    cols = [
        s.new_full((n,), init).scatter_reduce_(0, o, val, reduce=reduce)
        for val, init, reduce in ((xs, big, "amin"), (ys, big, "amin"), (xe, -1, "amax"), (ye, -1, "amax"))
    ]
    return torch.stack(cols, 1)
