from typing import Annotated

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice

# https://github.com/pytorch-labs/segment-anything-fast/blob/de861af9badd03e8b40f5e063d70754a3dc6b4f4/segment_anything_fast/utils/amg.py#L106-L141


@torch.compile(dynamic=True)
def _rle_decode_batch(padded_counts: Tensor, lengths: Tensor, num_pixels: int) -> Tensor:
    """Batched RLE decode kernel: pad counts, compute boundaries, scatter, cumsum.

    Args:
        padded_counts: [N, max_len] padded RLE counts.
        lengths: [N] number of valid counts per mask.
        num_pixels: h * w total pixels.

    Returns:
        [N, num_pixels] decoded flat masks as uint8.
    """
    n, max_len = padded_counts.shape

    # Batched cumsum over padded counts → boundary positions
    csum = torch.cumsum(padded_counts, dim=1)  # [N, max_len]

    # Boundaries are at csum[:, :-1] (all but last cumsum value)
    boundaries = csum[:, :-1]  # [N, max_len-1]

    # Valid mask: only positions within each mask's actual count length, excluding last
    col_idx = torch.arange(max_len - 1, device=padded_counts.device).unsqueeze(0)  # [1, max_len-1]
    valid = col_idx < (lengths.unsqueeze(1) - 1)  # [N, max_len-1]

    # Clamp boundary values to valid pixel range for scatter
    boundaries_clamped = boundaries.clamp(0, num_pixels - 1)

    # Batch indices for scatter into [N, num_pixels]
    signal = torch.zeros(n, num_pixels, device=padded_counts.device, dtype=torch.long)

    # Use valid mask to zero out invalid boundaries before scatter
    ones = valid.long()  # [N, max_len-1]
    signal.scatter_add_(1, boundaries_clamped, ones)

    # Batched cumsum + mod 2 → decode
    mask_flat = (torch.cumsum(signal, dim=1) % 2).to(torch.uint8)  # [N, num_pixels]

    return mask_flat


@torch.inference_mode()
def rleDecode(  # noqa: N802
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool = False,
) -> Annotated[tv.Mask, "H W N"]:
    """Decode binary masks encoded via RLE.

    Uses batched cumsum + scatter to decode all masks simultaneously.

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

    # Pad all count tensors to same length
    count_list = []
    for r in rles:
        counts = r.cnts.to(dtype=torch.long, device=src_device)
        if (counts < 0).any():
            raise RuntimeError("negative counts in RLE — invalid RLE mask representation")
        count_list.append(counts)

    lengths = torch.tensor([c.numel() for c in count_list], dtype=torch.long, device=src_device)
    padded_counts = pad_sequence(count_list, batch_first=True)  # [N, max_len]

    mask_flat = _rle_decode_batch(padded_counts, lengths, num_pixels)  # [N, num_pixels]

    # Reshape: RLE is column-major (w, h) then transpose to (h, w)
    data = mask_flat.view(n, w, h).permute(2, 1, 0).contiguous()  # [H, W, N]

    return tv.Mask(data, device=device, requires_grad=requires_grad)  # ty:ignore[no-matching-overload]
