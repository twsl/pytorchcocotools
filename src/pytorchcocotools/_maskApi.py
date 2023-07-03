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

    for i in range(n // 2):
        x = rle_tensor[i * 2]
        cnt = rle_tensor[i * 2 + 1]

        while cnt != 0:
            c = x & 31  # 0x1F = 31
            if cnt < 32:  # 0x20 = 32
                s.append(c + 48)
                cnt = 0
            else:
                s.append(c + 96)
                cnt -= 32

            x >>= 5

    return bytes(s)


def stringToRLE(s: bytes) -> torch.Tensor:
    rle_list = []
    x = 0
    cnt = 0

    for c in s:
        if c >= 128:
            cnt += (c - 128) * 32
        else:
            cnt += c
            rle_list.append(x)
            rle_list.append(cnt)
            x = 0
            cnt = 0
        x = (x << 5) | (c & 31)
    return torch.tensor(rle_list)


def rleFrString(rle_str: bytes) -> Tensor:
    s = rle_str
    n = len(s)
    cnts = [0] * n
    p = 0
    m = 0

    while p < n:
        x = 0
        k = 0
        more = True

        while more:
            c = s[p] - 48
            x |= (c & 31) << (5 * k)  # 0x1F = 31
            more = bool(c & 32)  # 0x20 = 32
            p += 1
            k += 1
            if not more and (c & 16):  # 0x10 = 16
                x |= -1 << (5 * k)

        if m > 2:
            x += cnts[m - 2]

        cnts[m] = x
        m += 1

    if len(cnts) % 2 != 0:
        cnts.append(0)

    return Tensor(cnts)


def stringToRLE_(s: bytes) -> Tensor:
    rle_tensor = []
    n = len(s)
    i = 0

    while i < n:
        c = s[i] - 48

        if c <= 31:  # If c < 32, it represents a count of 1-31
            x = c
            cnt = 1
        else:
            x = c & 31  # Extract the lower 5 bits representing the value
            cnt = c >> 5  # Shift right to get the count

            # Continue reading additional bytes if the count requires more than 5 bits
            while c >= 32:
                i += 1
                c = s[i] - 96
                x += (c & 31) << 5  # Extract the lower 5 bits and shift to the left
                cnt += c >> 5  # Shift right to get the count

        rle_tensor.append(x)
        rle_tensor.append(cnt)
        i += 1

    return torch.tensor(rle_tensor, dtype=torch.float)


def b(encoded_bytes: bytes) -> Tensor:
    """Decodes the encoded bytes back to the original RLE tensor."""
    rle_tensor = []
    x = 0
    cnt = 0

    for byte in encoded_bytes:
        c = byte - 48
        if c >= 48:  # For values encoded with cnt >= 32
            c -= 48
            cnt += 32

        x |= c << (cnt % 32)
        cnt += 6

        if cnt >= 32:
            rle_tensor.append(x)
            rle_tensor.append(cnt)
            x = 0
            cnt = 0

    return torch.tensor(rle_tensor, dtype=torch.float32)
