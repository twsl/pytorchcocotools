from typing import Annotated

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice

# https://github.com/pytorch-labs/segment-anything-fast/blob/de861af9badd03e8b40f5e063d70754a3dc6b4f4/segment_anything_fast/utils/amg.py#L106-L141


@torch.no_grad
def rleDecode(  # noqa: N802
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> Annotated[tv.Mask, "H W N"]:
    """Decode binary masks encoded via RLE.

    Uses vectorized repeat_interleave to decode all masks without Python loops
    over individual RLE segments.

    Args:
        rles: The run length encoded masks to decode.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The decoded binary masks in H×W×N format.
    """
    n = len(rles)
    h = rles[0].h
    w = rles[0].w
    num_pixels = h * w
    src_device = rles[0].cnts.device

    objs = []
    for r in rles:
        counts = r.cnts.to(dtype=torch.long, device=src_device)
        m = counts.shape[0]
        # Build alternating 0/1 values: index 0 is background (0), index 1 is foreground (1), etc.
        values = torch.arange(m, device=src_device, dtype=torch.uint8) % 2
        # Expand each value by its run length count
        mask_flat = torch.repeat_interleave(values, counts)
        # Truncate or pad to exact pixel count (handles rounding in RLE)
        if mask_flat.shape[0] > num_pixels:
            mask_flat = mask_flat[:num_pixels]
        elif mask_flat.shape[0] < num_pixels:
            mask_flat = torch.nn.functional.pad(mask_flat, (0, num_pixels - mask_flat.shape[0]))
        # Reshape: RLE is column-major (w, h) then transpose to (h, w)
        objs.append(mask_flat.view(w, h).t())

    data = torch.stack(objs, dim=-1)
    return tv.Mask(data, device=device, requires_grad=requires_grad)  # pyright: ignore[reportCallIssue]
