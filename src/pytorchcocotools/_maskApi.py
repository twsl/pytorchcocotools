from dataclasses import dataclass

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
    for index in unique_indices:
        values = transition_indices[torch.nonzero(transition_indices[:, 1] == index).squeeze(), 0]
        # cat_ind_tensor = torch.cat((zero_tensor, transition_indices, end))
        diff = torch.diff(values, prepend=start, dim=0)
        # run_steps = diff[::2]
        # run_lengths = torch.cat((diff[1::2], start))
        # rle = torch.cat([run_steps, run_lengths])
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


def rleMerge(R: RLEs, n: int, intersect: bool) -> RLEs:  # noqa: N802, N803
    """Compute union or intersection of encoded masks.

    Args:
        R: _description_
        n: _description_
        intersect: _description_

    Returns:
        _description_
    """
    if not R:
        return RLE()
    if len(R) == 1:
        return RLE(R[0].h, R[0].w, R[0].m, R[0].cnts)

    h, w, m = R[0].h, R[0].w, R[0].m
    cnts = R[0].cnts
    for i in range(1, len(R)):
        B = R[i]
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


def rleArea(R: RLEs, n: int) -> Tensor:  # noqa: N802, N803
    """Compute area of encoded masks.

    Args:
        R: _description_
        n: _description_

    Returns:
        _description_
    """
    a = [0] * n
    for i in range(n):
        for j in range(1, R[i].m, 2):
            a[i] += R[i].cnts[j]
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
        _description_
    """
    bb = [0] * (n * 4)
    for i in range(n):
        if R[i].m == 0:
            continue
        h, w, m = R[i].h, R[i].w, R[i].m
        m = (m // 2) * 2
        xs, ys, xe, ye = w, h, 0, 0
        cc = 0
        for j in range(m):
            cc += R[i].cnts[j]
            t = cc - (j % 2)
            y, x = t % h, t // h
            if j % 2 == 0:
                xp = x
            else:
                if xp < x:
                    ys, ye = 0, h - 1
            xs, xe = min(xs, x), max(xe, x)
            ys, ye = min(ys, y), max(ye, y)
        bb[4 * i : 4 * i + 4] = [xs, ys, xe - xs + 1, ye - ys + 1]
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
    scale = 5
    x, y = [int(scale * xy[j * 2] + 0.5) for j in range(k)], [int(scale * xy[j * 2 + 1] + 0.5) for j in range(k)]
    x.append(x[0])
    y.append(y[0])

    # compute rle encoding given y-boundary points
    a = sorted({x[i] * h + y[i] for i in range(k)})
    a.append(h * w)
    b = [a[0]]
    for i in range(1, len(a)):
        if a[i] > a[i - 1]:
            b.append(a[i] - a[i - 1])

    # Initialize RLE with the counts
    R = RLE(h=h, w=w, m=len(b), cnts=b)
    return R


def rleToString(R: RLE) -> bytes:  # noqa: N803, N802
    """Get compressed string representation of encoded mask.

    Args:
        R: _description_

    Note:
        Similar to LEB128 but using 6 bits/char and ascii chars 48-111.

    Returns:
        _description_
    """
    s = bytearray()
    cnts = R.cnts
    cnts = cnts.ceil().int()  # make sure it's integers

    for i in range(R.m):  # len(cnts)
        x = cnts[i]
        if i > 2:
            x -= cnts[i - 2]
        more = True
        while more:
            # take the 5 least significant bits of start point
            c = x & 0x1F  # 0x1f = 31
            # shift right by 5 bits as there are already read in
            x >>= 5
            # (c & 0x10) != 0 or x != 0
            more = bool(c & 0x10) if x != -1 else x != 0
            if more:
                c |= 0x20
            c += 48
            s.append(c)
    return bytes(s)


def rleFrString(s: bytes, h: int, w: int) -> RLE:  # noqa: N802
    """Convert from compressed string representation of encoded mask.

    Args:
        s: _description_
        h: _description_
        w: _description_

    Returns:
        _description_
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

    if len(cnts) % 2 != 0:
        cnts.append(0)

    return RLE(h, w, len(cnts), Tensor(cnts))
