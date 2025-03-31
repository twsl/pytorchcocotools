import torch
from torch import Tensor
from torchvision import tv_tensors as tv
from torchvision.transforms.v2 import functional as F  # noqa: N812

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice
from pytorchcocotools.internal.mask_api.rle_fr_poly import rleFrPoly
from pytorchcocotools.utils.poly import Polygon


@torch.no_grad
@torch.compile
def rleFrBbox(  # noqa: N802
    bb: tv.BoundingBoxes,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RLEs:
    """Convert bounding boxes to encoded masks.

    Args:
        bb: The bounding boxes.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The RLE encoded masks.
    """
    bb = tv.wrap(F.convert_bounding_box_format(bb, new_format=tv.BoundingBoxFormat.XYWH), like=bb)

    n = bb.shape[0]
    # Precompute the xy coordinates for all bounding boxes
    xs = bb[:, 0]
    ys = bb[:, 1]
    xe = xs + bb[:, 2]
    ye = ys + bb[:, 3]

    # Stack and reshape to get the xy tensor for all bounding boxes
    xy = torch.stack([xs, ys, xs, ye, xe, ye, xe, ys], dim=1).view(n, 4, 2)

    # Apply rleFrPoly (assumed to be vectorized) to each bounding box
    r = RLEs(
        [
            rleFrPoly(
                Polygon(xy[i], canvas_size=bb.canvas_size),  # pyright: ignore[reportCallIssue]
                device=device,
                requires_grad=requires_grad,
            )
            for i in range(n)
        ]
    )
    return r
