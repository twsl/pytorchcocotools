import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


def _rle_merge_two(
    cnts_a: list[int],
    cnts_b: list[int],
    intersect: bool,
) -> list[int]:
    """Scalar two-pointer merge of two RLE count lists.

    Directly mirrors the C reference algorithm: advance both run pointers
    simultaneously, consuming min(ca, cb) pixels per step.  The merged value
    is the union or intersection of the current foreground states va/vb.
    A new run is emitted whenever the merged value changes or both sequences
    are exhausted.

    This is O(R) in the total number of runs, with negligible fixed overhead —
    much cheaper than tensor dispatch for the small run counts typical in COCO.
    """
    len_a = len(cnts_a)
    len_b = len(cnts_b)
    if len_a == 0 or len_b == 0:
        return []

    ca = cnts_a[0]
    cb = cnts_b[0]
    v = va = vb = vp = 0
    a = b = 1  # next index to read
    cc = 0
    ct = 1
    result: list[int] = []

    while ct > 0:
        c = ca if ca < cb else cb  # min without function-call overhead
        cc += c
        ct = 0
        ca -= c
        if ca == 0 and a < len_a:
            ca = cnts_a[a]
            a += 1
            va ^= 1
        ct += ca
        cb -= c
        if cb == 0 and b < len_b:
            cb = cnts_b[b]
            b += 1
            vb ^= 1
        ct += cb
        vp = v
        v = 1 if (va and vb) else 0 if intersect else 1 if (va or vb) else 0
        if v != vp or ct == 0:
            result.append(cc)
            cc = 0

    return result


@torch.inference_mode()
def rleMerge(  # noqa: N802
    rles: RLEs,
    intersect: bool,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool = False,
) -> RLE:
    """Compute union or intersection of encoded masks.

    Uses a scalar two-pointer algorithm (mirrors the C reference) applied
    iteratively over N input masks.  RLE count tensors are converted to Python
    lists once and back once; all merging is done in pure Python loops over
    the run arrays — negligible overhead for the <50-run counts typical in COCO.

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

    cnts: list[int] = rles[0].cnts.tolist()
    for i in range(1, n):
        B = rles[i]  # noqa: N806
        if B.h != h or B.w != w:
            return RLE(0, 0, torch.tensor([]))
        cnts = _rle_merge_two(cnts, B.cnts.tolist(), intersect)

    return RLE(h, w, torch.tensor(cnts, dtype=torch.long, device=src_device))
