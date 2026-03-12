import torch

from pytorchcocotools.internal.entities import RLE, TorchDevice
from pytorchcocotools.utils.poly import Polygon


@torch.no_grad
# @torch.compile(dynamic=True)
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

    max_len_int = int(dxy.max().item()) + 1
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

    # Zero-merge: Python loop is faster than scatter_add for typical small arrays (n<400)
    if a.numel() == 0:
        b = torch.empty(0, dtype=torch.int32, device=device)
    else:
        a_list = a.tolist()
        b_list: list[int] = []
        j = 0
        while j < len(a_list):
            if a_list[j] > 0:
                b_list.append(a_list[j])
                j += 1
            else:
                j += 1
                if j < len(a_list) and b_list:
                    b_list[-1] += a_list[j]
                    j += 1
        b = (
            torch.tensor(b_list, dtype=torch.int32, device=device)
            if b_list
            else torch.empty(0, dtype=torch.int32, device=device)
        )

    # Initialize RLE with the counts
    r = RLE(h=h, w=w, cnts=b)
    return r
