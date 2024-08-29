import torch
from torch import Tensor
from torchvision.ops.boxes import nms

from pytorchcocotools.internal.entities import BB, RLE, Mask, RLEs
from pytorchcocotools.internal.mask_api.bb_iou import bbIou


# TODO: Note used in python api, call torch nms directly
def bbNms(dt: BB, n: int, thr: float) -> list[bool]:  # noqa: N802
    """Compute non-maximum suppression between bounding boxes.

    Args:
        dt: The detected bounding boxes (shape: [n, 4]).
        n: The number of detected bounding boxes.
        thr: The IoU threshold for non-maximum suppression.

    Returns:
        _description_
    """
    keep = [True] * n
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j]:
                    u = bbIou(dt[i], dt[j], [False])
                    if u[0].float() > thr:
                        keep[j] = False
    return keep
