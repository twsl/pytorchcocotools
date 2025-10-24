from typing import Annotated

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.no_grad
# @torch.compile
def rleEncode(  # noqa: N802
    mask: Annotated[tv.Mask, "N H W"],
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RLEs:
    """Encode binary masks using RLE.

    Args:
        mask: The binary masks to encode.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        _description_
    """
    n, h, w = mask.shape
    mask_p = mask.permute(0, 2, 1)
    flattened_mask = torch.flatten(mask_p, start_dim=1, end_dim=2).permute(1, 0)
    start_sentinel = torch.zeros((1, n), dtype=flattened_mask.dtype, device=mask.device)
    sentinel = torch.ones((1, n), dtype=flattened_mask.dtype, device=flattened_mask.device) * 2
    flat_tensor_with_sentinels = torch.cat([start_sentinel, flattened_mask, sentinel])

    # Optimized: Compute transitions once
    transitions = flat_tensor_with_sentinels[:-1, :] != flat_tensor_with_sentinels[1:, :]
    transition_indices = transitions.nonzero()

    # Optimized: Use list comprehension with tensor slicing
    zero = torch.zeros((1,), dtype=flattened_mask.dtype, device=mask.device)

    rles = []
    for index in range(n):
        # Optimized: Direct boolean indexing instead of nested nonzero
        mask_idx = transition_indices[:, 1] == index
        values = transition_indices[mask_idx, 0]
        diff = torch.diff(values, prepend=zero, dim=0)
        rles.append(RLE(h, w, diff))
    return RLEs(rles)
