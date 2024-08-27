import torch
from torch import Tensor

from pytorchcocotools.internal.entities import BB, RLE, Mask, RLEs


def rleToString(R: RLE) -> bytes:  # noqa: N803, N802
    """Get compressed string representation of encoded mask.

    Args:
        R: Run length encoded string mask.

    Note:
        Similar to LEB128 but using 6 bits/char and ascii chars 48-111.

    Returns:
        Byte string of run length encoded mask.
    """
    s = bytearray()
    cnts = R.cnts
    cnts = cnts.ceil().int()  # make sure it's integers

    for i in range(R.m):  # len(cnts)
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
