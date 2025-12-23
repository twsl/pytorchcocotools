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
    """Compute area of encoded masks using vectorized tensor operations.
    
    Uses fully vectorized approach when all RLEs have the same length,
    falling back to per-RLE vectorized sum for variable-length RLEs.

    Args:
        rles: The run length encoded masks.
        device: The desired device of the areas.
        requires_grad: Whether the areas require gradients.

    Returns:
        A tensor of areas of the encoded masks.
    """
    if not rles:
        return torch.tensor([], dtype=torch.int32, device=device, requires_grad=requires_grad if requires_grad else False)

    # Try fully vectorized approach if all RLEs have same length
    lengths = [len(rle.cnts) for rle in rles]
    if len(set(lengths)) == 1 and lengths[0] > 0:
        # All RLEs have same length - use fully vectorized approach
        # Stack all counts into a single tensor [num_rles, counts_per_rle]
        all_cnts = torch.stack([rle.cnts for rle in rles], dim=0)
        # Sum only odd indices (foreground pixels) along the counts dimension
        areas = all_cnts[:, 1::2].sum(dim=1).int()
    else:
        # Variable-length RLEs - fall back to per-RLE vectorized sum
        areas = torch.stack([torch.sum(rle.cnts[1::2]).int() for rle in rles])
    
    if device is not None:
        areas = areas.to(device)
    if requires_grad is not None:
        areas.requires_grad_(requires_grad)
    return areas
