import torch

from pytorchcocotools.internal.entities import RLE, TorchDevice
from pytorchcocotools.utils.poly import Polygon


@torch.inference_mode()
# @torch.compile — disabled: Polygon is a tensor subclass (like tv_tensors),
# causes dynamo recursion in __torch_function__. Fix expected in next PyTorch version.
def rleFrPoly(  # noqa: N802
    xy: Polygon,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool = False,
) -> RLE:
    """Convert polygon to encoded mask.

    Args:
        xy: The polygon vertices.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The RLE encoded mask.
    """
    h = xy.canvas_size[0]
    w = xy.canvas_size[1]
    device = xy.device
    # upsample and get discrete points densely along entire boundary
    scale = 5.0
    # Combine x/y scaling + wrap into one cat (was two separate cats)
    xy_int = (scale * xy + 0.5).int()
    xy_closed = torch.cat((xy_int, xy_int[0:1]))
    x = xy_closed[:, 0]
    y = xy_closed[:, 1]

    xs = x[:-1]
    xe = x[1:]
    ys = y[:-1]
    ye = y[1:]
    dx = torch.abs(xe - xs)
    dy = torch.abs(ys - ye)
    flip = ((dx >= dy) & (xs > xe)) | ((dx < dy) & (ys > ye))
    xs, xe = torch.where(flip, xe, xs), torch.where(flip, xs, xe)
    ys, ye = torch.where(flip, ye, ys), torch.where(flip, ys, ye)
    xy_cond = dx >= dy
    # Safe denominators guard against division by zero on axis-aligned edges.
    # torch.where evaluates both branches; clamping avoids silent inf/nan on GPU.
    s = torch.where(
        xy_cond,
        (ye - ys).to(dtype=torch.float64) / dx.clamp(min=1),
        (xe - xs).to(dtype=torch.float64) / dy.clamp(min=1),
    )  # double
    # Deduplicate: seq_lens and dxy are the same where-expression; compute once
    dxy = torch.where(xy_cond, dx, dy)
    seq_lens = dxy.unsqueeze(-1)

    max_len_int = int(dxy.max().clamp(min=0).to(dtype=torch.long)) + 1
    d = torch.arange(0, max_len_int, device=device).unsqueeze(0)
    d = d.expand(seq_lens.size(0), d.size(1))
    d_mask = d <= seq_lens

    t = torch.where(flip.unsqueeze(1), dxy.unsqueeze(1) - d, d)

    # Precompute s⊗t once; it was computed twice (for u and v) in the original
    xs2 = xs.unsqueeze(1)
    ys2 = ys.unsqueeze(1)
    cond2 = xy_cond.unsqueeze(1)
    s_t = s.unsqueeze(1) * t  # float64, computed once
    u = torch.where(cond2, t + xs2, (xs2 + s_t + 0.5).int())
    v = torch.where(cond2, (ys2 + s_t + 0.5).int(), t + ys2)

    u = u[d_mask]
    v = v[d_mask]

    # get points along y-boundary and downsample
    # Cache slices — each is referenced twice below
    u_next = u[1:]
    u_prev = u[:-1]
    v_next = v[1:]
    v_prev = v[:-1]

    changed = u_next != u_prev

    # Integer-domain boundary x coordinate before downsampling
    xd_int = torch.where(u_next < u_prev, u_next, u_next - 1)

    # Boundary checks entirely in integer space — avoids float floor/compare.
    # xd = (xd_int - 2) / 5.0; it is an integer iff xd_int ≡ 2 (mod 5).
    # range: xd ∈ [0, w-1] ↔ xd_int ∈ [2, 5*(w-1)+2]
    x_is_pixel = xd_int % 5 == 2
    x_in_range = (xd_int >= 2) & (xd_int <= 5 * (w - 1) + 2)
    full_mask = x_is_pixel & x_in_range & changed

    # Apply full_mask before any float work — compute only for surviving points
    xd_int_f = xd_int[full_mask]
    yd_int = torch.where(v_next < v_prev, v_next, v_prev)
    yd_int_f = yd_int[full_mask]

    # Compute final pixel coordinates as integers (no float needed).
    # xd guaranteed integer since xd_int ≡ 2 (mod 5) → exact division.
    # yd formula: ceil(clamp((yd_int-2)/5, 0, h)) = clamp(⌈(yd_int-2)/5⌉, 0, h)
    #           = clamp((yd_int+2) // 5, 0, h)  using ceiling = (n+d-1)//d for d=5
    xd_coord = (xd_int_f - 2).div(5, rounding_mode="floor")
    yd_coord = ((yd_int_f + 2).div(5, rounding_mode="floor")).clamp(0, h)

    # compute rle encoding given y-boundary points — all-integer computation
    a = xd_coord * h + yd_coord
    a = torch.cat((a, a.new_tensor([h * w])))

    sorted_a = torch.sort(a).values
    a = torch.diff(sorted_a, prepend=sorted_a.new_zeros(1)).to(dtype=torch.int32)

    # Zero-merge: remove zero-count runs by merging adjacent same-parity runs.
    # Fully vectorized so torch.compile can trace the entire function without graph breaks.
    #
    # Algorithm translated to tensors:
    #   "claimed[j] = (a[j-1]==0) & ~claimed[j-1]" recurrence →
    #   an element is  "claimed" if the element immediately to its left is zero AND
    #   that zero was not itself claimed.
    #
    # We compute claimed via two observations:
    #  (a) A zero is "active" (claims its right neighbour) iff it is NOT claimed.
    #  (b) Because active zeros eat pairs (zero, next), within a run of consecutive
    #      zeros the alternating pattern means only odd-indexed zeros (0-based) in
    #      the run are active.
    # Using this we can compute the claimed mask in O(1) tensor passes.
    if a.numel() == 0:
        b = torch.empty(0, dtype=torch.int32, device=device)
    else:
        # claimed[j] = (a[j-1] == 0) & ~claimed[j-1], claimed[0]=False
        # Compute via: within each run of consecutive zeros, alternate True/False.
        zm = a == 0  # zero mask [n]
        # shifted: is the previous element a zero?
        prev_zm = torch.cat([zm.new_zeros(1), zm[:-1]])  # [n]
        # Within a consecutive-zero run, the position parity determines claimed.
        # run_start[j] = True if this is the beginning of a consecutive-zero run.
        run_start = prev_zm & ~torch.cat([prev_zm.new_zeros(1), prev_zm[:-1]])  # [n]
        # offset within each zero-run (0-indexed): cumsum(prev_zm) - cumsum at run start
        cum = torch.cumsum(prev_zm.long(), dim=0)
        # cumsum value at the start of each run (broadcast across the run):
        cum_at_start = torch.cumsum(run_start.long() * cum, dim=0)
        # offset[j] = cum[j] - cum[j at start of its run] = position within run
        # We need cum_at_start to contain, for each j, the cum value frozen at the
        # most recent run_start. Use a scatter + cummax trick:
        frozen = cum * run_start  # non-zero only at run starts
        # Propagate forward: frozen_propagated[j] = max(frozen[0..j]) among run starts
        frozen_propagated = torch.cummax(frozen, dim=0).values  # [n]
        offset_in_run = cum - frozen_propagated  # 0-indexed position in current zero-run
        # claimed[j] = prev_zm[j] AND (offset_in_run[j] is even, i.e. % 2 == 0)
        # (because the 0th element in a run is active, so offset 0 → next is claimed=True)
        # Note: offset_in_run is 0 for the first member of a run that follows a zero,
        # 1 for the second, etc.
        claimed = prev_zm & ((offset_in_run % 2) == 0)  # [n]

        # "active zeros": zero positions that are NOT claimed
        active_zero = zm & ~claimed  # [n]

        # Positions that are "skipped" (not output as a regular element):
        # - Active zero positions (they contribute nothing, their value is 0)
        # - Claimed positions (their value is added to the previous regular element)
        skipped = active_zero | claimed  # [n]
        regular = ~skipped  # positions that become output elements

        # Output segment for each position: regular positions start new segments,
        # claimed positions belong to the PREVIOUS regular segment.
        seg_id = torch.cumsum(regular.long(), dim=0) - 1  # 0-indexed segment id, -1 for
        # elements before the first regular (these have regular=False and seg_id<0 → claimed)
        # Claimed elements at the very start (before any regular element) have seg_id=-1,
        # but b_list is empty so they must be dropped—use torch.clamp to handle.
        seg_id_clamped = seg_id.clamp(min=0)

        # Number of output segments — use tensor max to avoid .item() graph break.
        # seg_id is -1 for elements before the first regular, so max(seg_id)+1 = num_segs.
        num_segs = seg_id.max().clamp(min=-1).to(dtype=torch.long) + 1

        # Allocate output and scatter_add all contributing elements in one pass.
        # regular elements → their own segment; claimed elements → preceding segment.
        # Using max(num_segs, 1) ensures we always have at least size-1 output
        # so scatter_add is safe even when num_segs==0 (output will be sliced to 0).
        out = torch.zeros(num_segs.clamp(min=1), dtype=a.dtype, device=device)
        # Regular elements contribute to their own segment
        out.scatter_add_(0, seg_id_clamped[regular], a[regular])
        # Claimed elements merge into the preceding regular segment
        valid_claimed = claimed & (seg_id >= 0)
        out.scatter_add_(0, seg_id_clamped[valid_claimed], a[valid_claimed])
        b = out[:num_segs]

    # Initialize RLE with the counts
    r = RLE(h=h, w=w, cnts=b)
    return r
