from typing import Annotated

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.inference_mode()
def rleEncode(  # noqa: N802
    mask: Annotated[tv.Mask, "N H W"],
    *,
    device: TorchDevice | None = None,
    requires_grad: bool = False,
) -> RLEs:
    """Encode binary masks using RLE.

    Uses vectorized transition detection with batch-first layout to avoid
    expensive argsort. Transitions are computed on [N, H*W] so nonzero()
    returns pairs already sorted by (mask_id, position).

    Args:
        mask: The binary masks to encode.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        Run length encoded masks.
    """
    n, h, w = mask.shape
    hw = h * w

    # Flatten to column-major order: [N, H*W]
    mask_data = mask.as_subclass(Tensor)
    flat = mask_data.permute(0, 2, 1).reshape(n, hw)

    # Sentinels: 0 at start, 2 at end to detect first/last transitions
    start_sentinel = torch.zeros((n, 1), dtype=flat.dtype, device=mask_data.device)
    end_sentinel = torch.full((n, 1), 2, dtype=flat.dtype, device=mask_data.device)
    with_sentinels = torch.cat([start_sentinel, flat, end_sentinel], dim=1)  # [N, hw+2]

    # Transitions: [N, hw+1] — batch-first so nonzero is sorted by (mask, position)
    transitions = with_sentinels[:, :-1] != with_sentinels[:, 1:]
    transition_indices = transitions.nonzero()  # [K, 2]: (mask_id, position)

    if transition_indices.shape[0] == 0:
        return RLEs([RLE(h, w, torch.tensor([hw], dtype=torch.long, device=mask.device)) for _ in range(n)])

    mask_ids = transition_indices[:, 0]
    positions = transition_indices[:, 1]

    # Count transitions per mask — already sorted by mask_id, no argsort needed
    counts_per_mask = torch.zeros(n, dtype=torch.long, device=mask.device)
    counts_per_mask.scatter_add_(0, mask_ids, torch.ones_like(mask_ids, dtype=torch.long))

    # Split into per-mask groups (positions are already in order within each group)
    split_sizes = counts_per_mask.tolist()
    groups = torch.split(positions, split_sizes)

    zero = torch.zeros((1,), dtype=positions.dtype, device=mask.device)
    rles = []
    for i in range(n):
        if split_sizes[i] == 0:
            rles.append(RLE(h, w, torch.tensor([hw], dtype=torch.long, device=mask.device)))
        else:
            values = groups[i]
            diff = torch.diff(values, prepend=zero, dim=0)
            rles.append(RLE(h, w, diff))
    return RLEs(rles)
