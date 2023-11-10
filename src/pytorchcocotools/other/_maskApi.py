from typing import Union

import torch
from torch import Tensor

R = dict
Rs = list[R]
RLES = tuple[list[Tensor], list[tuple[int, int]]]  # list of steps, lengths
MASK = Tensor  # HWN
SIZES = list[tuple[int, int]]


def rleEncode(mask: MASK) -> tuple[RLES, SIZES]:
    h, w, n = mask.shape
    flattened_mask = torch.flatten(mask, start_dim=0, end_dim=-2)  # numpy default is hwn
    zero_tensor = torch.zeros((1, n), device=mask.device)
    # Find the indices where the mask transitions from 0 to 1 and from 1 to 0
    transitions = torch.cat((zero_tensor, flattened_mask, zero_tensor))
    transitions = transitions[:-1, :] != transitions[1:, :]
    transition_indices = transitions.nonzero()

    # end = torch.ones((1,), device=mask.device) * flattened_mask.shape[0]
    unique_indices = torch.unique(transition_indices[:, 1])
    start = torch.zeros((1,), device=mask.device)

    rles = []
    sizes = []
    for index in unique_indices:
        values = transition_indices[torch.nonzero(transition_indices[:, 1] == index).squeeze(), 0]
        # cat_ind_tensor = torch.cat((zero_tensor, transition_indices, end))
        diff = torch.diff(values, prepend=start, dim=0)
        run_steps = diff[::2]
        run_lengths = torch.cat((diff[1::2], start))
        rle = torch.cat([run_steps, run_lengths])
        rles.append(rle)
        sizes += [(h, w)]
    return rles, sizes


def rleDecode(rles: RLES, sizes: SIZES) -> MASK:
    objs = []
    for r, size in zip(rles, sizes):
        counts = r
        # Calculate the number of pixels in the binary mask
        num_pixels = size[0] * size[1]
        # Create a binary mask tensor of zeros
        mask_tensor = torch.zeros(num_pixels, dtype=torch.uint8)
        # calculate absolute counts from relative counts
        counts = torch.cumsum(torch.tensor(counts, dtype=torch.long), dim=0)
        # Create pairs of starting and ending indices from the counts
        pairs = torch.split(counts, 2)
        # Create a list of the indices of the 1s in the mask
        indices_list = [torch.arange(start, end) for start, end in pairs]
        # Set the corresponding pixels in the mask to 1 using vectorized indexing
        mask_tensor[torch.cat(indices_list)] = 1
        # Reshape the 1D tensor into a 2D binary mask tensor
        mask_tensor = mask_tensor.reshape(size[0], size[1])
        objs.append(mask_tensor)
    return torch.stack(objs, dim=-1)


def rleMerge(rles: RLES, sizes: SIZES, intersect: bool = False) -> Tensor:
    # Initialize the output tensor with zeros
    merged_mask = torch.sum(rleDecode(rles, sizes), dim=0)
    # If the intersection flag is set, set the pixels that are not present in all masks to 0
    if intersect:
        merged_mask[merged_mask < len(rles)] = 0
    # Reshape the 1D tensor into a 2D binary mask tensor
    merged_mask = merged_mask.reshape(sizes[0][0], sizes[0][1])
    return merged_mask

    #     cnts = np.zeros(R[0].m, dtype=np.uint32)
    #     h, w, m = R[0].h, R[0].w, R[0].m
    #     if n == 0:
    #         M.rleInit(0, 0, 0, 0)
    #         return
    #     if n == 1:
    #         M.rleInit(h, w, m, R[0].cnts)
    #         return
    #     for a in range(m):
    #         cnts[a] = R[0].cnts[a]
    #     for i in range(1, n):
    #         B = R[i]
    #         if B.h != h or B.w != w:
    #             h = w = m = 0
    #             break
    #         A = M.rleInit(h, w, m, cnts)
    #         ca = A.cnts[0]
    #         cb = B.cnts[0]
    #         v = va = vb = 0
    #         m = 0
    #         a = b = 1
    #         cc = 0
    #         ct = 1
    #         while ct > 0:
    #             c = min(ca, cb)
    #             cc += c
    #             ct = 0
    #             ca -= c
    #             if not ca and a < A.m:
    #                 ca = A.cnts[a]
    #                 va = not va
    #             ct += ca
    #             cb -= c
    #             if not cb and b < B.m:
    #                 cb = B.cnts[b]
    #                 vb = not vb
    #             ct += cb
    #             vp = v
    #             if intersect:
    #                 v = va and vb
    #             else:
    #                 v = va or vb
    #             if v != vp or ct == 0:
    #                 cnts[m] = cc
    #                 m += 1
    #                 cc = 0
    pass


def rleArea(rleObjs: Rs) -> list[int]:
    return None


def rleIoU():
    pass


def rleNms(dt: Tensor, n: int, keep: bool, thr: float):
    pass


def bbIou(dt: Tensor, gt: Tensor, m: int, n: int, iscrowd: bool, o: float):
    pass


def bbNms(dt: Tensor, n: int, keep, thr: float):
    pass


def rleToBbox(R, bb: Tensor, n):
    pass


def rleFrBbox(R, bb: Tensor, h: int, w: int, n: int):
    pass


def rleFrPoly(R, xy: float, k: int, h: int, w: int):
    pass


def rleToString(rle_tensor: Tensor) -> bytes:  # noqa: N802
    """Similar to LEB128 but using 6 bits/char and ascii chars 48-111."""
    s = bytearray()
    n = len(rle_tensor)
    rle_tensor = rle_tensor.ceil().int()

    for i in range(n):
        x = rle_tensor[i]
        if i > 2:
            x -= rle_tensor[i - 2]
        more = True
        while more:
            # take the 5 least significant bits of start point
            c = x & 0x1F  # 0x1f = 31
            # shift right by 5 bits as there are already read in
            x >>= 5
            more = bool(c & 0x10) if x != -1 else x != 0  # (c & 0x10) != 0 or x != 0
            if more:
                c |= 0x20
            c += 48
            s.append(c)
    return bytes(s)


def rleFrString(rle_str: bytes) -> Tensor:  # noqa: N802
    """Similar to LEB128 but using 6 bits/char and ascii chars 48-111."""
    m = 0
    p = 0
    s = rle_str
    n = len(s)
    cnts = []
    while p < n:
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
        cnts.append(x) #cnts[m] = x
        m += 1

    if len(cnts) % 2 != 0:
        cnts.append(0)

    return Tensor(cnts)


##################################################

def rle_to_dict(R: dict[str, int]) -> bytes:
    """Similar to LEB128 but using 6 bits/char and ascii chars 48-111."""
    m = R["m"]
    cnts = R["cnts"]
    s = bytearray()
    for i in range(m):
        x = cnts[i]
        if i > 2:
            x -= cnts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (c & 0x10) != 0 if x != -1 else x != 0
            if more:
                c |= 0x20
            c += 48
            s.append(c)
    return bytes(s)


def rle_fr_string(s: bytes, h: int, w: int) -> dict[str, list[int]]:
    m = len(s)
    p = 0
    cnts = []
    while p < m:
        x = 0
        k = 0
        more = True
        while more:
            c = s[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = (c & 0x20) != 0
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return {"h": h, "w": w, "m": len(cnts), "cnts": cnts}


def rle_init(R: dict[str, list[int]], h: int, w: int) -> None:
    R["h"] = h
    R["w"] = w
    R["m"] = len(R["cnts"])
