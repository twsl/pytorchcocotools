from typing import Annotated

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice

# https://github.com/pytorch-labs/segment-anything-fast/blob/de861af9badd03e8b40f5e063d70754a3dc6b4f4/segment_anything_fast/utils/amg.py#L106-L141


# TODO: vectorize
def rleDecode(  # noqa: N802
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> Annotated[tv.Mask, "H W N"]:
    """Decode binary masks encoded via RLE.

    Args:
        rles: The run length encoded masks to decode.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        _description_
    """
    objs = []
    for r in rles:
        counts = r.cnts
        h = r.h
        w = r.w
        size = [h, w]
        # Calculate the number of pixels in the binary mask
        num_pixels = size[0] * size[1]
        # Create a binary mask tensor of zeros
        mask_tensor = torch.zeros(num_pixels, dtype=torch.uint8, device=counts.device)
        # calculate absolute counts from relative counts
        counts = torch.cumsum(counts.to(dtype=torch.long), dim=0)
        # Create pairs of starting and ending indices from the counts
        pairs = list(torch.split(counts, 2))
        m = len(counts) // 2
        # Create a list of the indices of the 1s in the mask
        indices_list = [torch.arange(start=int(start.int()), end=int(end.int())) for start, end in pairs[:m]]
        # Set the corresponding pixels in the mask to 1 using vectorized indexing
        mask_tensor[torch.cat(indices_list)] = 1
        # Reshape the 1D tensor into a 2D binary mask tensor
        mask_tensor = mask_tensor.view(w, h).t()
        objs.append(mask_tensor)
    data = torch.stack(objs, dim=-1)
    return tv.Mask(data, device=device, requires_grad=requires_grad)  # pyright: ignore[reportCallIssue]
