import torch
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.mask_api.bb_iou import bbIou


@torch.no_grad
# @torch.compile
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
