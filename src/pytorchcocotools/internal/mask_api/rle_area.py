import torch
from torch import Tensor

from pytorchcocotools.internal.entities import BB, RLE, Mask, RLEs


def rleArea(R: RLEs, n: int) -> list[int]:  # noqa: N802, N803
    """Compute area of encoded masks.

    Args:
        R: _description_
        n: _description_

    Returns:
        A list of areas of the encoded masks.
    """
    a = [int(torch.sum(R[i].cnts[1 : R[i].m : 2]).int()) for i in range(n)]
    return a
