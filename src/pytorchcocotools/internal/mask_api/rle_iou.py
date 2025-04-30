import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs
from pytorchcocotools.internal.mask_api.bb_iou import bbIou
from pytorchcocotools.internal.mask_api.rle_area import rleArea
from pytorchcocotools.internal.mask_api.rle_to_bbox import rleToBbox


@torch.no_grad
# @torch.compile
def rleIou(dt: RLEs, gt: RLEs, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between masks.

    Args:
        dt: The RLE encoded detection masks.
        gt: The RLE encoded ground truth masks.
        iscrowd: The crowd label for each ground truth mask.

    Returns:
        The intersection over union between the masks.
    """
    db = rleToBbox(dt)
    gb = rleToBbox(gt)
    m = len(dt)
    n = len(gt)
    o = bbIou(db, gb, iscrowd)
    for g in range(n):
        for d in range(m):
            if o[d, g] > 0:
                crowd = iscrowd is not None and iscrowd[g]
                if dt[d].h != gt[g].h or dt[d].w != gt[g].w:
                    o[d, g] = -1
                    continue
                ka = len(dt[d].cnts)  # m
                kb = len(gt[g].cnts)  # m
                ca = dt[d].cnts[0]
                cb = gt[g].cnts[0]
                va = False
                vb = False
                a = 1
                b = 1
                u = 0
                i = 0
                ct = 1
                while ct > 0:
                    c = torch.min(ca, cb).clone()
                    if va or vb:
                        u += c
                        if va and vb:
                            i += c
                    ct = 0
                    ca -= c
                    if not ca and a < ka:
                        ca = dt[d].cnts[a]
                        a += 1
                        va = not va
                    ct += ca
                    cb -= c
                    if not cb and b < kb:
                        cb = gt[g].cnts[b]
                        b += 1
                        vb = not vb
                    ct += cb
                if i == 0:
                    u = 1
                elif crowd:
                    u = rleArea([dt[d]])[0]
                o[d, g] = i / u
    return o
