import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.no_grad
def rleMerge(  # noqa: N802
    rles: RLEs,
    intersect: bool,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RLE:
    """Compute union or intersection of encoded masks.

    Uses two-pointer merge with pure Python ints to avoid per-iteration
    tensor allocation overhead. Pairwise merges all input RLEs.

    Args:
        rles: The masks to merge.
        intersect: Whether to compute the intersection.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The merged mask.
    """
    n = len(rles)
    if not rles or n == 0:
        return RLE(0, 0, Tensor())
    if n == 1:
        return rles[0]

    h, w = rles[0].h, rles[0].w
    src_device = rles[0].cnts.device

    cnts_list: list[int] = rles[0].cnts.tolist()
    for i in range(1, n):
        B = rles[i]  # noqa: N806
        if B.h != h or B.w != w:
            return RLE(0, 0, torch.tensor([]))

        cnts_b: list[int] = B.cnts.tolist()
        ka = len(cnts_list)
        kb = len(cnts_b)
        ca = int(cnts_list[0])
        cb = int(cnts_b[0])
        v = False
        va = False
        vb = False
        a = 1
        b = 1
        cc = 0
        ct = 1
        cnts_out: list[int] = []
        while ct > 0:
            c = min(ca, cb)
            cc += c
            ct = 0
            ca -= c
            if not ca and a < ka:
                ca = int(cnts_list[a])
                a += 1
                va = not va
            cb -= c
            if not cb and b < kb:
                cb = int(cnts_b[b])
                b += 1
                vb = not vb
            ct += ca
            ct += cb
            vp = v
            v = va and vb if intersect else va or vb
            if v != vp or ct == 0:
                cnts_out.append(cc)
                cc = 0
        cnts_list = cnts_out

    return RLE(h, w, torch.tensor(cnts_list, device=src_device))
