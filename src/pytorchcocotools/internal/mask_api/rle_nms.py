from pytorchcocotools.internal.entities import RLEs
from pytorchcocotools.internal.mask_api.rle_iou import rleIou


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
    if n <= 1:
        return [True] * n
    keep = [True] * n
    # Precompute the full n×n IoU matrix in one batched call instead of making
    # O(n²/2) separate 1×1 calls.  For n≥4 (n*n≥10) rleIou uses the vectorised
    # prefix-sum path; for smaller n the scalar two-pointer path handles all
    # pairs at once without per-call Python overhead.
    iou_matrix = rleIou(dt, dt, [False] * n)  # [n, n]
    iou_list = iou_matrix.tolist()
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j] and iou_list[i][j] > thr:
                    keep[j] = False
    return keep
