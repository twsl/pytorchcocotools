import torch
from torch import Tensor

from pytorchcocotools.internal.entities import BB, RLE, Mask, RLEs
from pytorchcocotools.internal.mask_api.rle_fr_poly import rleFrPoly


# TODO: fix input tensor form
def rleFrBbox(bb: BB, h: int, w: int) -> RLEs:  # noqa: N802
    """Convert bounding boxes to encoded masks.

    Args:
        bb: The bounding boxes.
        h: The height of the image.
        w: The width of the image.

    Returns:
        The RLE encoded masks.
    """
    n = bb.shape[0]
    # Precompute the xy coordinates for all bounding boxes
    xs = bb[:, 0]
    ys = bb[:, 1]
    xe = xs + bb[:, 2]
    ye = ys + bb[:, 3]

    # Stack and reshape to get the xy tensor for all bounding boxes
    xy = torch.stack([xs, ys, xs, ye, xe, ye, xe, ys], dim=1).view(n, 8)

    # Apply rleFrPoly (assumed to be vectorized) to each bounding box
    r = RLEs([rleFrPoly(xy[i], 4, h, w) for i in range(n)])
    return r
