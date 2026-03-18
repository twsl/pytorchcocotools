from typing import Annotated

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice

# https://github.com/pytorch-labs/segment-anything-fast/blob/de861af9badd03e8b40f5e063d70754a3dc6b4f4/segment_anything_fast/utils/amg.py#L106-L141


@torch.inference_mode()
def rleDecode(  # noqa: N802
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool = False,
) -> Annotated[tv.Mask, "H W N"]:
    """Decode binary masks encoded via RLE.

    Uses cumsum + scatter to decode each mask without data-dependent shapes.

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
        if (counts < 0).any():
            raise RuntimeError("negative counts in RLE — invalid RLE mask representation")
        csum = torch.cumsum(counts, dim=0)
        boundaries = csum[:-1]
        signal = torch.zeros(num_pixels, device=src_device, dtype=torch.long)
        signal.scatter_add_(0, boundaries, torch.ones_like(boundaries))
        mask_flat = (torch.cumsum(signal, dim=0) % 2).to(torch.uint8)
        objs.append(mask_flat.view(w, h).t())

    data = torch.stack(objs, dim=-1)
    return tv.Mask(data, device=device, requires_grad=requires_grad)  # ty:ignore[no-matching-overload]
