import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


def rleToBbox(  # noqa: N802,
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> tv.BoundingBoxes:
    """Get bounding boxes surrounding encoded masks.

    Args:
        rles: The RLE encoded masks.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        List of bounding boxes in format [x y w h]
    """
    n = len(rles)
    device = rles[0].cnts.device
    bb = torch.zeros((n, 4), dtype=torch.int32, device=device)
    for i in range(n):
        if len(rles[i].cnts) == 0:  # m
            continue
        h, _w, m = rles[i].h, rles[i].w, len(rles[i].cnts)
        m = (m // 2) * 2

        cc = torch.cumsum(rles[i].cnts[:m], dim=0)
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

    return tv.BoundingBoxes(bb, format=tv.BoundingBoxFormat.XYWH, canvas_size=(rles[0].h, rles[0].w))  # pyright: ignore[reportCallIssue]
