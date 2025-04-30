import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice
from pytorchcocotools.utils.poly import Polygon


@torch.no_grad
# @torch.compile
def rleFrPoly(  # noqa: N802
    xy: Polygon,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RLE:
    """Convert polygon to encoded mask.

    Args:
        xy: The polygon vertices.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The RLE encoded mask.
    """
    # k = xy.num_coordinates
    h = xy.canvas_size[0]
    w = xy.canvas_size[1]
    device = xy.device
    # upsample and get discrete points densely along entire boundary
    scale = 5.0
    x = ((scale * xy[:, 0]) + 0.5).int()
    x = torch.cat((x, x[0:1]))
    y = ((scale * xy[:, 1]) + 0.5).int()
    y = torch.cat((y, y[0:1]))

    max_diff = torch.maximum(torch.abs(torch.diff(x)), torch.abs(torch.diff(y))) + 1
    m = torch.sum(max_diff).int()

    xs = x[:-1]
    xe = x[1:]
    ys = y[:-1]
    ye = y[1:]
    dx = torch.abs(xe - xs)
    dy = torch.abs(ys - ye)
    flip = ((dx >= dy) & (xs > xe)) | ((dx < dy) & (ys > ye))
    _xs = torch.where(flip, xe, xs)
    _xe = torch.where(flip, xs, xe)
    _ys = torch.where(flip, ye, ys)
    _ye = torch.where(flip, ys, ye)
    xs = _xs
    xe = _xe
    ys = _ys
    ye = _ye
    xy_cond = dx >= dy
    s = torch.where(xy_cond, (ye - ys).to(dtype=torch.float64) / dx, (xe - xs).to(dtype=torch.float64) / dy)  # double
    seq_lens = torch.where(xy_cond, dx, dy).unsqueeze(-1)

    max_len = torch.max(seq_lens)
    max_len_int = int(max_len.int().item()) + 1
    d = torch.arange(0, max_len_int, device=device).unsqueeze(0)
    d = d.expand(seq_lens.size(0), d.size(1))
    d_mask = d <= seq_lens

    dxy = torch.where(xy_cond, dx, dy)
    t = torch.where(flip.unsqueeze(1), dxy.unsqueeze(1) - d, d)
    # t = torch.clamp(t, 0, d)

    u = torch.where(xy_cond.unsqueeze(1), t + xs.unsqueeze(1), (xs.unsqueeze(1) + s.unsqueeze(1) * t + 0.5).int())
    v = torch.where(xy_cond.unsqueeze(1), (ys.unsqueeze(1) + s.unsqueeze(1) * t + 0.5).int(), t + ys.unsqueeze(1))

    u = u[d_mask]
    v = v[d_mask]
    # assert u.size(0) == m  # noqa: S101

    # get points along y-boundary and downsample
    changed = u[1:] != u[:-1]
    xd = torch.where(u[1:] < u[:-1], u[1:], u[1:] - 1)
    xd = (xd + 0.5) / scale - 0.5

    fl_mask = torch.floor(xd) != xd
    sm_mask = xd < 0
    lg_mask = xd > w - 1
    or_mask = torch.logical_or(fl_mask, torch.logical_or(sm_mask, lg_mask))
    cont_mask = torch.logical_not(or_mask)

    full_mask = torch.logical_and(cont_mask, changed)

    yd = torch.where(v[1:] < v[:-1], v[1:], v[:-1])
    yd = (yd + 0.5) / scale - 0.5
    yd = torch.clamp(yd, 0, h)
    yd = torch.ceil(yd)

    xd = xd[full_mask]
    yd = yd[full_mask]
    # compute rle encoding given y-boundary points
    a = xd * h + yd
    a = torch.cat((a, torch.tensor([h * w], device=device)))

    sorted, _ = torch.sort(a)

    a = torch.diff(sorted, prepend=torch.tensor([0], device=device)).to(dtype=torch.int)

    b = torch.zeros_like(a, device=device)
    j, m = 0, 0
    while j < len(a):
        if a[j] > 0:
            b[m] = a[j]
            m += 1
            j += 1
        else:
            j += 1
            if j < len(a):
                b[m - 1] += a[j]
                j += 1

    b = b[:m]

    # Initialize RLE with the counts
    r = RLE(h=h, w=w, cnts=b)
    return r
