import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, TorchDevice
from pytorchcocotools.utils.poly import Polygon


@torch.inference_mode()
@torch.compile(mode="reduce-overhead")
def _rle_fr_poly_core(xy: Tensor, h: int, w: int) -> Tensor:
    """Core polygon→RLE computation on plain Tensors.

    Outlined from rleFrPoly so torch.compile can trace without
    hitting Polygon (TVTensor subclass) dynamo recursion.
    """
    device = xy.device

    # 1. Upsample polygon to 5× integer grid
    xy_int = (5.0 * xy + 0.5).int()
    xy_closed = torch.cat((xy_int, xy_int[:1]))

    # Edge start/end coordinates
    xs, xe = xy_closed[:-1, 0], xy_closed[1:, 0]
    ys, ye = xy_closed[:-1, 1], xy_closed[1:, 1]
    dx = (xe - xs).abs()
    dy = (ys - ye).abs()

    # Direction normalization
    flip = ((dx >= dy) & (xs > xe)) | ((dx < dy) & (ys > ye))
    xs, xe = torch.where(flip, xe, xs), torch.where(flip, xs, xe)
    ys, ye = torch.where(flip, ye, ys), torch.where(flip, ys, ye)

    xy_cond = dx >= dy
    dxy = torch.where(xy_cond, dx, dy)
    # Merged slope computation — single where for numerator and denominator
    s = torch.where(xy_cond, ye - ys, xe - xs).double() / dxy.clamp(min=1)

    # 2. Dense point expansion via 2D grid + mask
    d = torch.arange(dxy.max() + 1, device=device).unsqueeze(0).expand(dxy.size(0), -1)  # ty:ignore[no-matching-overload]
    d_mask = d <= dxy.unsqueeze(-1)

    t = torch.where(flip.unsqueeze(1), dxy.unsqueeze(1) - d, d)

    # 3. Interpolate u, v for each dense point
    s_t = s.unsqueeze(1) * t
    cond2 = xy_cond.unsqueeze(1)
    xs2 = xs.unsqueeze(1)
    ys2 = ys.unsqueeze(1)

    u = torch.where(cond2, t + xs2, (xs2 + s_t + 0.5).int())
    v = torch.where(cond2, (ys2 + s_t + 0.5).int(), t + ys2)

    u = u[d_mask]
    v = v[d_mask]

    # 4. Boundary crossing detection + pixel grid filtering (single fused mask)
    u_next = u[1:]
    u_prev = u[:-1]
    xd_int = torch.where(u_next < u_prev, u_next, u_next - 1)
    full_mask = (u_next != u_prev) & (xd_int % 5 == 2) & (xd_int >= 2) & (xd_int <= 5 * (w - 1) + 2)

    # 5. Extract surviving boundary points → pixel coordinates (integer only)
    xd_int_f = xd_int[full_mask]
    # Compute min(v_next, v_prev) first, then mask once
    yd_int_f = torch.minimum(v[1:], v[:-1])[full_mask]
    xd_coord = (xd_int_f - 2).div(5, rounding_mode="floor")
    yd_coord = ((yd_int_f + 2).div(5, rounding_mode="floor")).clamp(0, h)

    # 6–7. RLE encoding with duplicate-boundary cancellation.
    positions = torch.cat((xd_coord * h + yd_coord, xd_coord.new_tensor([h * w])))
    sorted_positions = positions.sort().values
    unique_positions, counts = torch.unique_consecutive(sorted_positions, return_counts=True)
    keep = (counts & 1) == 1
    keep[-1] = True
    return torch.diff(unique_positions[keep], prepend=unique_positions.new_zeros(1)).to(torch.int32)


@torch.inference_mode()
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
    xy_tensor = xy.as_subclass(Tensor)
    cnts = _rle_fr_poly_core(xy_tensor, h, w)
    return RLE(h=h, w=w, cnts=cnts)
