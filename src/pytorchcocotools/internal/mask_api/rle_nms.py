import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs
from pytorchcocotools.internal.mask_api.rle_iou import rleIou


@torch.no_grad
@torch.compile
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
