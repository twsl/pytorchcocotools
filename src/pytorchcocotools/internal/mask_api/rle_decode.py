import torch
from torch import Tensor

from pytorchcocotools.internal.entities import BB, RLE, Mask, RLEs


# TODO: vectorize
def rleDecode(R: RLEs, n: int) -> Mask:  # noqa: N802, N803
    """Decode binary masks encoded via RLE.

    Args:
        R: _description_
        n: _description_

    Returns:
        _description_
    """
    objs = []
    for r in R:
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
    return torch.stack(objs, dim=-1)
