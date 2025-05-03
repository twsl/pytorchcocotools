import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.no_grad
@torch.compile
def rleArea(  # noqa: N802
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> list[int]:
    """Compute area of encoded masks.

    Args:
        rles: The run length encoded masks.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        A list of areas of the encoded masks.
    """
    a = [int(torch.sum(rle.cnts[1::2]).int().item()) for rle in rles]
    return a
