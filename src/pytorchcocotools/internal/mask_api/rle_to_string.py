import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RleObj, RleObjs, RLEs, TorchDevice


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
    # make sure it's integers
    cnts = cnts.ceil().int()
    
    # Optimized: Convert to Python list once instead of repeated .item() calls
    cnts_list = cnts.tolist()

    for i in range(len(cnts_list)):
        x = cnts_list[i]
        if i > 2:
            x -= cnts_list[i - 2]
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


def rleToStringBatch(  # noqa: N802
    rles: RLEs,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RleObjs:
    """Get compressed string representation of encoded mask.

    Args:
        rles: Run length encoded string masks.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Note:
        Similar to LEB128 but using 6 bits/char and ascii chars 48-111.

    Returns:
        Byte string of run length encoded mask.
    """
    results = []
    for rle in rles:
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
        results.append(
            RleObj(
                size=[rle.h, rle.w],
                counts=bytes(s),
            )
        )
    return results
