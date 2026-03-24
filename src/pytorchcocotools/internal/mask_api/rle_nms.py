import torch

from pytorchcocotools.internal.entities import RLEs
from pytorchcocotools.internal.mask_api.rle_iou import rleIou


# TODO: Not used in python api
@torch.inference_mode()
def rleNms(dt: RLEs, n: int, thr: float) -> list[bool]:  # noqa: N802
    """Compute non-maximum suppression between bounding masks.

    Args:
        dt: The detected masks
        n: The number of detected masks.
        thr: The IoU threshold for non-maximum suppression.

    Returns:
        The mask indices to keep.
    """
    if n <= 1:
        return [True] * n
    # Precompute the full n×n IoU matrix in one batched call.
    iou_matrix = rleIou(dt[:n], dt[:n], [False] * n)  # [n, n]

    # Vectorized greedy suppression (matches the original sequential algorithm):
    # For each detection i (in order), suppress all j > i with IoU > thr if i is kept.
    keep_t = torch.ones(n, dtype=torch.bool, device=iou_matrix.device)
    for i in range(n):
        if keep_t[i]:
            # Suppress all j > i where iou > thr
            suppress = iou_matrix[i, i + 1 :] > thr
            keep_t[i + 1 :] = keep_t[i + 1 :] & ~suppress
    return keep_t.tolist()
