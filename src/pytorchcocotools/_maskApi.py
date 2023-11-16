from dataclasses import dataclass
import math

import torch
from torch import Tensor


class BB(Tensor):
    pass


class RLE:
    def __init__(self, h: int = 0, w: int = 0, m: int = 0, cnts: Tensor = None):
        """Internal run length encoded representation.

        Args:
            h: The mask height. Defaults to 0.
            w: The mask width. Defaults to 0.
            m: The number of rle entries. Defaults to 0.
            cnts: The rle entries. Defaults to None.
        """
        self.h = h
        self.w = w
        self.m = m
        self.cnts = cnts if cnts is not None else torch.zeros(m, dtype=torch.int32)


class RLEs(list[RLE]):
    def __init__(self, rles: list[RLE], n: int = None):
        self.n = n if n is not None else len(rles) if len(rles) > 0 else 0
        super().__init__(rles)


class Mask(Tensor):
    """# hxwxn binary mask, in column-major order.

    Args:
        Tensor: _description_
    """

    pass


class Masks(list[Mask]):
    def __init__(self, masks: list[Mask], h: int = None, w: int = None, n: int = None):
        self.h = h if h is not None else masks[0].shape[0] if len(masks) > 0 else 0
        self.w = w if w is not None else masks[0].shape[1] if len(masks) > 0 else 0
        self.n = n if n is not None else len(masks) if len(masks) > 0 else 0
        super().__init__(masks)


@dataclass
class RleObj:
    size: tuple[int, int]
    counts: bytes

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __delitem__(self, key):
        return delattr(self, key)

    def __iter__(self):
        for key in self.__dataclass_fields__:
            yield key, getattr(self, key)

    def __len__(self):
        return len(self.__dataclass_fields__)


class RleObjs(list[RleObj]):
    pass


def rleEncode(mask: Mask, h: int, w: int, n: int) -> RLEs:  # noqa: N802
    """Encode binary masks using RLE.

    Args:
        mask: _description_
        h: _description_
        w: _description_
        n: _description_

    Returns:
        _description_
    """
    H, W, N = mask.shape
    flattened_mask = torch.flatten(mask, start_dim=0, end_dim=-2)
    start_sentinel = torch.zeros((1, N), dtype=flattened_mask.dtype, device=mask.device)
    end = torch.ones((1, N), dtype=flattened_mask.dtype, device=mask.device) * flattened_mask.shape[0]
    sentinel = torch.ones((1, N), dtype=flattened_mask.dtype, device=flattened_mask.device) * 2
    flat_tensor_with_sentinels = torch.cat([start_sentinel, flattened_mask, sentinel])

    transitions = flat_tensor_with_sentinels[:-1, :] != flat_tensor_with_sentinels[1:, :]
    transition_indices = transitions.nonzero()

    unique_indices = torch.unique(transition_indices[:, 1])
    zero = torch.zeros((1,), dtype=flattened_mask.dtype, device=mask.device)

    rles = []
    for index in unique_indices:
        values = transition_indices[torch.nonzero(transition_indices[:, 1] == index).squeeze(), 0]
        # not adding append=end, because thats how pycocotools does it
        # Results in the possibility of an uneven number of values
        diff = torch.diff(values, prepend=zero, dim=0)
        rle = diff
        rles.append(RLE(h, w, len(rle), rle))
    return RLEs(rles)


def rleDecode(R: RLEs, n: int) -> Mask:  # noqa: N802, N803
    """Decode binary masks encoded via RLE.

    Args:
        R: _description_
        n: _description_

    Returns:
        _description_
    """
    objs = []
    for r in R:
        counts = r.cnts
        h = r.h
        w = r.w
        size = [h, w]
        # Calculate the number of pixels in the binary mask
        num_pixels = size[0] * size[1]
        # Create a binary mask tensor of zeros
        mask_tensor = torch.zeros(num_pixels, dtype=torch.uint8, device=counts.device)
        # calculate absolute counts from relative counts
        counts = torch.cumsum(counts.to(dtype=torch.long), dim=0)
        # Create pairs of starting and ending indices from the counts
        pairs = list(torch.split(counts, 2))
        m = len(counts) // 2
        # Create a list of the indices of the 1s in the mask
        indices_list = [torch.arange(start, end) for start, end in pairs[:m]]
        # Set the corresponding pixels in the mask to 1 using vectorized indexing
        mask_tensor[torch.cat(indices_list)] = 1
        # Reshape the 1D tensor into a 2D binary mask tensor
        mask_tensor = mask_tensor.reshape(size[0], size[1])
        objs.append(mask_tensor)
    return torch.stack(objs, dim=-1)


def rleMerge(Rs: RLEs, n: int, intersect: bool) -> RLEs:  # noqa: N802, N803
    """Compute union or intersection of encoded masks.

    Args:
        R: _description_
        n: _description_
        intersect: _description_

    Returns:
        _description_
    """
    if not Rs:
        return RLE()
    n = len(Rs)
    if n == 1:
        return Rs[0]

    h, w, m = Rs[0].h, Rs[0].w, Rs[0].m
    cnts = Rs[0].cnts
    for i in range(1, n):
        B = Rs[i]
        if B.h != h or B.w != w:
            return RLE()  # Return an empty RLE if dimensions don't match

        A = RLE(h, w, m, cnts)
        ca, cb = A.cnts[0], B.cnts[0]
        v = va = vb = 0
        m = a = b = 0
        cc = 0
        ct = 1
        cnts = []
        while ct > 0:
            c = min(ca, cb)
            cc += c
            ct = 0
            ca -= c
            if not ca and a < len(A.cnts) - 1:
                a += 1
                ca = A.cnts[a]
                va = not va
            ct += ca

            cb -= c
            if not cb and b < len(B.cnts) - 1:
                b += 1
                cb = B.cnts[b]
                vb = not vb
            ct += cb

            vp = v
            v = va and vb if intersect else va or vb
            if v != vp or ct == 0:
                cnts.append(cc)
                cc = 0

    return RLE(h, w, len(cnts), cnts)


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


def rleIou(dt: RLEs, gt: RLEs, m: int, n: int, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between masks.

    Args:
        dt: _description_
        gt: _description_
        m: _description_
        n: _description_
        iscrowd: _description_

    Returns:
        _description_
    """
    o = torch.zeros((m * n,), dtype=torch.float32)
    db = [None] * (m * 4)
    gb = [None] * (n * 4)
    db = rleToBbox(dt, m)
    gb = rleToBbox(gt, n)
    o = bbIou(db, gb, m, n, iscrowd)
    for g in range(n):
        for d in range(m):
            if o[g * m + d] > 0:
                crowd = iscrowd is not None and iscrowd[g]
                if dt[d].h != gt[g].h or dt[d].w != gt[g].w:
                    o[g * m + d] = -1
                    continue
                ka, kb = dt[d].m, gt[g].m
                ca, cb = dt[d].cnts[0], gt[g].cnts[0]
                va, vb = 0, 0
                a, b = 1, 1
                u = i = 0
                ct = 1
                while ct > 0:
                    c = min(ca, cb)
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
                    u = rleArea(dt[d], 1)[0]
                o[g * m + d] = i / u


def rleNms(dt: RLE, n: int, thr: float) -> list[bool]:  # noqa: N802
    """Compute non-maximum suppression between bounding masks.

    Args:
        dt: _description_
        n: _description_
        thr: _description_

    Returns:
        _description_
    """
    keep = [True] * n
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j]:
                    u = rleIou([dt[i]], [dt[j]], 1, 1, None)
                    if u[0].float() > thr:
                        keep[j] = False
    return keep


def bbIou(dt: BB, gt: BB, m: int, n: int, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between bounding boxes.

    Args:
        dt: _description_
        gt: _description_
        m: _description_
        n: _description_
        iscrowd: _description_

    Returns:
        _description_
    """
    o = torch.zeros((m * n,), dtype=torch.float32)
    for g in range(n):
        G = gt[g * 4 : (g + 1) * 4]
        ga = G[2] * G[3]
        crowd = iscrowd is not None and iscrowd[g]
        for d in range(m):
            D = dt[d * 4 : (d + 1) * 4]
            da = D[2] * D[3]
            o[g * m + d] = 0
            w = min(D[2] + D[0], G[2] + G[0]) - max(D[0], G[0])
            if w <= 0:
                continue
            h = min(D[3] + D[1], G[3] + G[1]) - max(D[1], G[1])
            if h <= 0:
                continue
            i = w * h
            u = da if crowd else da + ga - i
            o[g * m + d] = i / u
    return o


def bbNms(dt: BB, n: int, thr: float) -> list[bool]:  # noqa: N802
    """Compute non-maximum suppression between bounding boxes.

    Args:
        dt: _description_
        n: _description_
        thr: _description_

    Returns:
        _description_
    """
    keep = [True] * n
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j]:
                    u = bbIou(dt[i * 4 : (i + 1) * 4], dt[j * 4 : (j + 1) * 4], 1, 1, None)
                    if u[0].float() > thr:
                        keep[j] = False
    return keep


def rleToBbox(R: RLEs, n: int) -> BB:  # noqa: N802, N803
    """Get bounding boxes surrounding encoded masks.

    Args:
        R: _description_
        n: _description_

    Returns:
        List of bounding boxes in format [x y w h]
    """
    device = R[0].cnts.device
    bb = torch.zeros((n, 4), dtype=torch.int32, device=device)
    for i in range(n):
        if R[i].m == 0:
            continue
        h, w, m = R[i].h, R[i].w, R[i].m
        m = (m // 2) * 2

        cc = torch.cumsum(R[i].cnts[:m], dim=0)
        # Calculate x, y coordinates
        t = cc - torch.arange(m) % 2
        y = t % h
        x = t // h

        xs = torch.min(x[0::2])
        xe = torch.max(x[1::2])
        ys = torch.min(y[0::2])
        ye = torch.max(y[1::2])

        # Adjust for full height in case of full column runs
        full_col_mask = x[0::2] < x[1::2]
        if torch.any(full_col_mask):
            ys = torch.full_like(ys, 0)
            ye = torch.full_like(ye, (h - 1))

        bb[i] = torch.stack([xs, ys, xe - xs + 1, ye - ys + 1])
    return bb


def rleFrBbox(bb: BB, h: int, w: int, n: int) -> RLEs:  # noqa: N802
    """Convert bounding boxes to encoded masks.

    Args:
        bb: _description_
        h: _description_
        w: _description_
        n: _description_

    Returns:
        _description_
    """
    R = [None] * n
    for i in range(n):
        xs, xe = bb[4 * i], bb[4 * i] + bb[4 * i + 2]
        ys, ye = bb[4 * i + 1], bb[4 * i + 1] + bb[4 * i + 3]
        xy = [xs, ys, xs, ye, xe, ye, xe, ys]
        R[i] = rleFrPoly(xy, 4, h, w)
    return R


def rleFrPoly(xy: Tensor, k: int, h: int, w: int) -> RLE:  # noqa: N802
    """Convert polygon to encoded mask.

    Args:
        xy: _description_
        k: _description_
        h: _description_
        w: _description_

    Returns:
        _description_
    """
    # upsample and get discrete points densely along entire boundary
    scale = 5.0
    x = [int(scale * xy[j * 2] + 0.5) for j in range(k)] + [int(scale * xy[0] + 0.5)]
    y = [int(scale * xy[j * 2 + 1] + 0.5) for j in range(k)] + [int(scale * xy[1] + 0.5)]
    m = 0
    for j in range(k):
        m += max(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1

    u, v = [], []
    for j in range(k):
        xs, xe, ys, ye = x[j], x[j + 1], y[j], y[j + 1]
        dx, dy = abs(xe - xs), abs(ys - ye)
        flip = (dx >= dy and xs > xe) or (dx < dy and ys > ye)
        if flip:
            xs, xe, ys, ye = xe, xs, ye, ys
        s = (ye - ys) / dx if dx >= dy else (xe - xs) / dy
        for d in range(dx + 1 if dx >= dy else dy + 1):
            t = dx - d if flip else d
            if dx >= dy:
                u.append(xs + t)
                v.append(int(ys + s * t + 0.5))
            else:
                v.append(ys + t)
                u.append(int(xs + s * t + 0.5))

    # Downsample and compute RLE encoding
    k = len(u)
    x, y = [], []
    for j in range(1, k):
        if u[j] != u[j - 1]:
            xd = u[j] if u[j] < u[j - 1] else u[j] - 1
            xd = (xd + 0.5) / scale - 0.5
            if not (0 <= xd < w):
                continue
            yd = v[j] if v[j] < v[j - 1] else v[j - 1]
            yd = (yd + 0.5) / scale - 0.5
            yd = max(0, min(yd, h))
            x.append(int(xd))
            y.append(int(math.ceil(yd)))

    a = [x[j] * h + y[j] for j in range(len(x))]
    a.append(h * w)
    a.sort()
    p = 0
    for j in range(len(a)):
        t = a[j]
        a[j] -= p
        p = t

    b, j, m = [], 0, 0
    while j < len(a):
        if a[j] > 0:
            b.append(a[j])
            m += 1
            j += 1
        else:
            j += 1
            if j < len(a):
                b[m - 1] += a[j]
                j += 1

    # Initialize RLE with the counts
    R = RLE(h=h, w=w, m=len(b), cnts=torch.tensor(b, dtype=torch.int))
    return R


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


def rleFrString(s: bytes, h: int, w: int) -> RLE:  # noqa: N802
    """Convert from compressed string representation of encoded mask.

    Args:
        s: Byte string of run length encoded mask.
        h: Height of the encoded mask.
        w: Width of the encoded mask.

    Returns:
        Run length encoded mask.
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

    return RLE(h, w, len(cnts), Tensor(cnts))
