import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.inference_mode()
def rleArea(  # noqa: N802
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool = False,
) -> list[int]:
    """Compute area of encoded masks.

    Args:
        rles: The run length encoded masks.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        A list of areas of the encoded masks.
    """
    if not rles:
        return []
    # Fast path when all cnts are the same length: stack into a 2D tensor and sum odd columns
    lengths = [r.cnts.numel() for r in rles]
    if lengths and all(l == lengths[0] for l in lengths) and lengths[0] > 0:
        src_device = rles[0].cnts.device
        stacked = torch.stack([r.cnts for r in rles], dim=0)  # [N, max_len]
        return stacked[:, 1::2].sum(dim=1).tolist()
    # General path: variable length cnts (list comprehension, one .item() per RLE)
    return [int(rle.cnts[1::2].sum().item()) for rle in rles]
