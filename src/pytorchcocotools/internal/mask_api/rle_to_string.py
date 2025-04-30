import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.no_grad
# @torch.compile
def rleToString(  # noqa: N802
    rle: RLE,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> bytes:
    """Get compressed string representation of encoded mask.

    Args:
        rle: Run length encoded string mask.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Note:
        Similar to LEB128 but using 6 bits/char and ascii chars 48-111.

    Returns:
        Byte string of run length encoded mask.
    """
    s = bytearray()
    cnts = rle.cnts
    cnts = cnts.ceil().int()  # make sure it's integers

    for i in range(len(rle.cnts)):  # len(cnts)
        x = int(cnts[i])  # make sure its not a reference
        if i > 2:
            x -= int(cnts[i - 2])
        more = True
        while more:
            # take the 5 least significant bits of start point
            c = x & 0x1F  # 0x1f = 31
            # shift right by 5 bits as there are already read in
            x >>= 5
            # (c & 0x10) != 0 or x != 0
            more = x != -1 if bool(c & 0x10) else x != 0
            if more:
                c |= 0x20
            c += 48
            s.append(c)
    return bytes(s)
