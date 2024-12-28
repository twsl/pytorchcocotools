import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


def rleFrString(  # noqa: N802
    s: bytes,
    h: int,
    w: int,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RLE:
    """Convert from compressed string representation of encoded mask.

    Args:
        s: Byte string of run length encoded mask.
        h: Height of the encoded mask.
        w: Width of the encoded mask.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The RLE encoded mask.
    """
    m = 0
    p = 0
    cnts = []
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = s[p] - 48
            x |= (c & 0x1F) << (5 * k)  # 0x1F = 31
            more = bool(c & 0x20)  # 0x20 = 32
            p += 1
            k += 1
            if not more and bool(c & 0x10):  # 0x10 = 16
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)  # cnts[m] = x
        m += 1

    # don't do this as pycocotools also ignores this
    # if len(cnts) % 2 != 0:
    #     cnts.append(0)

    result = torch.tensor(cnts, device=device, requires_grad=requires_grad if requires_grad else False)

    # uneven number of values means we cant reshape
    # result = result.view(-1, 2)
    return RLE(h, w, result)
