import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.mask_api.bb_iou import bbIou


@torch.no_grad
@torch.compile
# TODO: Note used in python api, call torch nms directly
def bbNms(dt: tv.BoundingBoxes, thr: float) -> list[bool]:  # noqa: N802
    """Compute non-maximum suppression between bounding boxes.

    Args:
        dt: The detected bounding boxes (shape: [n, 4]).
        thr: The IoU threshold for non-maximum suppression.

    Returns:
        _description_
    """
    n = dt.shape[0]
    keep = [True] * n
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j]:
                    u = bbIou(tv.wrap(dt[i], like=dt), tv.wrap(dt[j], like=dt), [False])
                    if u[0].float() > thr:
                        keep[j] = False
    return keep


@torch.no_grad
@torch.compile
def bbNmsBatch(dt: tv.BoundingBoxes, thr: float) -> Tensor:  # noqa: N802
    """Compute non-maximum suppression using vectorized operations.

    Args:
        dt: The detected bounding boxes (shape: [n, 4]).
        thr: The IoU threshold for non-maximum suppression.

    Returns:
        A boolean tensor indicating which boxes to keep (shape: [n]).
    """
    n = dt.shape[0]
    if n == 0:
        return torch.tensor([], dtype=torch.bool, device=dt.device)

    # Compute IoU matrix for all pairs in one vectorized operation
    iou_matrix = bbIou(dt, dt, [False] * n)
    
    # Create upper triangular mask (excluding diagonal) for suppression
    # This ensures we only consider pairs where j > i
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dt.device), diagonal=1)
    
    # Find all pairs exceeding threshold in upper triangle
    suppress_pairs = (iou_matrix > thr) & triu_mask
    
    # A box should be suppressed if ANY earlier box suppresses it
    # suppress_pairs[i, j] = True means box i suppresses box j
    # We want to mark j as suppressed if any i < j suppresses it
    suppressed = suppress_pairs.any(dim=0)
    
    # Keep boxes that are not suppressed
    keep = ~suppressed

    return keep
