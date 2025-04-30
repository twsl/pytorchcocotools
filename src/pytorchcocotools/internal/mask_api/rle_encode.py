import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.no_grad
# @torch.compile
# TODO: vectorize
def rleEncode(  # noqa: N802
    mask: tv.Mask,
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
    # torch.ones((1, n), dtype=flattened_mask.dtype, device=mask.device) * flattened_mask.shape[0] # TODO: ???
    sentinel = torch.ones((1, n), dtype=flattened_mask.dtype, device=flattened_mask.device) * 2
    flat_tensor_with_sentinels = torch.cat([start_sentinel, flattened_mask, sentinel])

    transitions = flat_tensor_with_sentinels[:-1, :] != flat_tensor_with_sentinels[1:, :]
    transition_indices = transitions.nonzero()

    unique_indices = torch.unique(transition_indices[:, 1])
    zero = torch.zeros((1,), dtype=flattened_mask.dtype, device=mask.device)

    rles = []
    for index in unique_indices:
        values = transition_indices[torch.nonzero(transition_indices[:, 1] == index).squeeze(), 0]
        # not adding append=end, because thats how pycocotools does it
        # Results in the possibility of an uneven number of values
        diff = torch.diff(values, prepend=zero, dim=0)
        rle = diff
        rles.append(RLE(h, w, rle))
    return RLEs(rles)
