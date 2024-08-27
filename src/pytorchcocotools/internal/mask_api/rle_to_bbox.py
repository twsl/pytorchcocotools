import torch
from torch import Tensor

from pytorchcocotools.internal.entities import BB, RLE, Mask, RLEs


def rleToBbox(R: RLEs, n: int) -> BB:  # noqa: N802, N803
    """Get bounding boxes surrounding encoded masks.

    Args:
        R: The RLE encoded masks.
        n: The number of encoded masks.

    Returns:
        List of bounding boxes in format [x y w h]
    """
    device = R[0].cnts.device
    bb = torch.zeros((n, 4), dtype=torch.int32, device=device)
    for i in range(n):
        if R[i].m == 0:
            continue
        h, _w, m = R[i].h, R[i].w, R[i].m
        m = (m // 2) * 2

        cc = torch.cumsum(R[i].cnts[:m], dim=0)
        # Calculate x, y coordinates
        t = cc - torch.arange(m) % 2
        y = t % h
        x = t // h

        xs = torch.min(x[0::2])
        xe = torch.max(x[1::2])
        ys = torch.min(y[0::2])
        ye = torch.max(y[1::2])

        # Adjust for full height in case of full column runs
        full_col_mask = x[0::2] < x[1::2]
        if torch.any(full_col_mask):
            ys = torch.full_like(ys, 0)
            ye = torch.full_like(ye, (h - 1))

        bb[i] = torch.stack([xs, ys, xe - xs + 1, ye - ys + 1])
    return bb
