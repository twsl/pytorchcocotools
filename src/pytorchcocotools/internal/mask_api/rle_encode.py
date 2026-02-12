from typing import Annotated

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.no_grad
def rleEncode(  # noqa: N802
    mask: Annotated[tv.Mask, "N H W"],
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RLEs:
    """Encode binary masks using RLE.

    Uses vectorized transition detection and per-mask grouping via
    pre-sorted indices to avoid repeated boolean scans.

    Args:
        mask: The binary masks to encode.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        Run length encoded masks.
    """
    n, h, w = mask.shape
    # Transpose to column-major layout and flatten
    mask_p = mask.permute(0, 2, 1)
    flattened_mask = torch.flatten(mask_p, start_dim=1, end_dim=2).permute(1, 0)
    start_sentinel = torch.zeros((1, n), dtype=flattened_mask.dtype, device=mask.device)
    sentinel = torch.ones((1, n), dtype=flattened_mask.dtype, device=flattened_mask.device) * 2
    flat_tensor_with_sentinels = torch.cat([start_sentinel, flattened_mask, sentinel])

    # Compute transitions once for all masks
    transitions = flat_tensor_with_sentinels[:-1, :] != flat_tensor_with_sentinels[1:, :]
    transition_indices = transitions.nonzero()  # (K, 2): [position, mask_index]

    # transition_indices is already sorted by (position, mask_index), but we need
    # to group by mask_index. Since nonzero() returns in row-major order and
    # transitions has shape (positions, masks), indices are sorted by position first.
    # Sort by mask_index to group them.
    if transition_indices.shape[0] == 0:
        return RLEs([RLE(h, w, torch.zeros(1, dtype=torch.long, device=mask.device)) for _ in range(n)])

    # Group by mask index using split_sizes from counts
    mask_ids = transition_indices[:, 1]
    positions = transition_indices[:, 0]

    # Count transitions per mask
    counts_per_mask = torch.zeros(n, dtype=torch.long, device=mask.device)
    counts_per_mask.scatter_add_(0, mask_ids, torch.ones_like(mask_ids, dtype=torch.long))

    # Sort by mask index, then by position within each mask
    sort_order = torch.argsort(mask_ids, stable=True)
    sorted_positions = positions[sort_order]

    # Split into per-mask groups
    split_sizes = counts_per_mask.tolist()
    groups = torch.split(sorted_positions, split_sizes)

    zero = torch.zeros((1,), dtype=sorted_positions.dtype, device=mask.device)
    rles = []
    for i in range(n):
        if split_sizes[i] == 0:
            rles.append(RLE(h, w, torch.tensor([h * w], dtype=torch.long, device=mask.device)))
        else:
            values = groups[i]
            diff = torch.diff(values, prepend=zero, dim=0)
            rles.append(RLE(h, w, diff))
    return RLEs(rles)
