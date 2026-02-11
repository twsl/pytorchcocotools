import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs
from pytorchcocotools.internal.mask_api.rle_iou import rleIou


@torch.no_grad
# @torch.compile
# TODO: Note used in python api
def rleNms(dt: RLEs, n: int, thr: float) -> list[bool]:  # noqa: N802
    """Compute non-maximum suppression between bounding masks.

    Args:
        dt: The detected masks
        n: The number of detected masks.
        thr: The IoU threshold for non-maximum suppression.

    Returns:
        The mask indices to keep.
    """
    keep = [True] * n
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j]:
                    u = rleIou([dt[i]], [dt[j]], [False])
                    if u[0].float() > thr:
                        keep[j] = False
    return keep


@torch.no_grad
# @torch.compile
def rleNmsBatch(dt: RLEs, n: int, thr: float) -> Tensor:  # noqa: N802
    """Compute non-maximum suppression using vectorized operations.

    Args:
        dt: The detected masks
        n: The number of detected masks.
        thr: The IoU threshold for non-maximum suppression.

    Returns:
        A boolean tensor indicating which masks to keep (shape: [n]).
    """
    if n == 0:
        return torch.tensor([], dtype=torch.bool)

    # Compute IoU matrix for all pairs in one batched operation
    iou_matrix = rleIou(dt, dt, [False] * n)
    
    # Create upper triangular mask (excluding diagonal)
    device = iou_matrix.device
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    
    # Find all pairs exceeding threshold in upper triangle
    suppress_pairs = (iou_matrix > thr) & triu_mask
    
    # A mask should be suppressed if ANY earlier mask suppresses it
    suppressed = suppress_pairs.any(dim=0)
    
    # Keep masks that are not suppressed
    keep = ~suppressed

    return keep
