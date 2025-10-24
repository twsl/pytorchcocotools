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


@torch.no_grad
@torch.compile
def rleAreaBatch(  # noqa: N802
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Compute area of encoded masks in a batched manner.

    Args:
        rles: The run length encoded masks.
        device: The desired device of the areas.
        requires_grad: Whether the areas require gradients.

    Returns:
        A tensor of areas of the encoded masks.
    """
    if not rles:
        return torch.tensor([], dtype=torch.int32, device=device, requires_grad=requires_grad if requires_grad else False)
    
    # For vectorization, we need to handle variable-length RLEs
    # We'll still iterate but use vectorized sum within each RLE
    areas = torch.stack([torch.sum(rle.cnts[1::2]).int() for rle in rles])
    if device is not None:
        areas = areas.to(device)
    if requires_grad is not None:
        areas.requires_grad_(requires_grad)
    return areas
