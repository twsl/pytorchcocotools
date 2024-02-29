from dataclasses import dataclass

import torch
from torch import Tensor


class BB(Tensor):
    pass


class RLE:
    def __init__(self, h: int = 0, w: int = 0, m: int = 0, cnts: torch.Tensor = None):
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
class RleObj(dict):
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
    mask_p = mask.permute(2, 1, 0)
    flattened_mask = torch.flatten(mask_p, start_dim=1, end_dim=2).permute(1, 0)
    start_sentinel = torch.zeros((1, n), dtype=flattened_mask.dtype, device=mask.device)
    torch.ones((1, n), dtype=flattened_mask.dtype, device=mask.device) * flattened_mask.shape[0]
    sentinel = torch.ones((1, n), dtype=flattened_mask.dtype, device=flattened_mask.device) * 2
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
        mask_tensor = mask_tensor.view(w, h).t()
        objs.append(mask_tensor)
    return torch.stack(objs, dim=-1)


def rleMerge(Rs: RLEs, n: int, intersect: bool) -> RLEs:  # noqa: N802, N803
    """Compute union or intersection of encoded masks.

    Args:
        Rs: _description_
        n: _description_
        intersect: _description_

    Returns:
        _description_
    """
    if not Rs or len({r.w for r in Rs}) != 1 or len({r.w for r in Rs}) != 1:
        return RLE()  # Return an empty RLE if empty or dimensions don't match
    n = len(Rs)
    if n == 1:
        return Rs

    # Rs[0].cnts.device
    h, w = Rs[0].h, Rs[0].w

    # se_inds = torch.cat([torch.cumsum(r.cnts[:-1], 0).reshape(-1, 2) for r in Rs])
    # seq_lens = (se_inds[:, 1] - se_inds[:, 0]).unsqueeze(-1)
    # max_len = torch.max(seq_lens)
    # d = torch.arange(0, max_len, device=device).unsqueeze(0)  # +1 maxlen?
    # d = d.expand(seq_lens.size(0), d.size(1))
    # d_mask = d <= seq_lens

    # inds = d + se_inds[:, 0].unsqueeze(1)
    # selected_inds = inds[d_mask]
    # unq_inds, counts = torch.unique(selected_inds, sorted=True, return_counts=True)
    # id_mask = counts >= (n if intersect else 1)
    # merged_ids = unq_inds[id_mask]

    # zero = torch.zeros((1,), device=device)
    # rle_max = torch.tensor([h * w], device=device)
    # diffs = torch.diff(merged_ids, prepend=zero, append=rle_max)

    # one_indices = torch.nonzero(diffs == 1).view(-1)
    # one_ind_diffs = torch.diff(one_indices)

    # start_indices = one_indices[:-1][torch.diff(one_indices) > 1]
    # end_indices = one_indices[1:][torch.diff(one_indices) > 1] - 1

    # # Handle the case when the last element is 1
    # if diffs[-1] == 1:
    #     start_indices = torch.cat([start_indices, one_indices[-1].unsqueeze(0)])
    #     end_indices = torch.cat([end_indices, one_indices[-1].unsqueeze(0)])

    # run_ends = torch.nonzero(diffs != 1, as_tuple=True)[0] + 1

    # # Create tensors to store run lengths and values
    # run_lengths = torch.cat([run_ends.new_tensor([run_ends[0]]), run_ends[1:] - run_ends[:-1]])
    # run_values = merged_ids[run_ends]

    # [skip,used,skip,used,skip]

    # return run_values, run_lengths

    h, w, m = Rs[0].h, Rs[0].w, Rs[0].m
    cnts = Rs[0].cnts.clone()
    for i in range(1, n):
        B = Rs[i]  # noqa: N806
        if B.h != h or B.w != w:
            return RLE()  # Return an empty RLE if dimensions don't match

        A = RLE(h, w, m, cnts)  # noqa: N806
        ca = A.cnts[0].clone()
        cb = B.cnts[0].clone()
        v = False
        va = False
        vb = False
        a = 1
        b = 1
        cc = 0
        ct = 1
        cnts = []
        while ct > 0:
            c = min(ca, cb).clone()
            cc += c
            ct = 0
            ca -= c
            if not ca and a < A.m:
                ca = A.cnts[a].clone()
                a += 1
                va = not va
            ct += ca
            cb -= c
            if not cb and b < B.m:
                cb = B.cnts[b].clone()
                b += 1
                vb = not vb
            ct += cb
            vp = v
            v = va and vb if intersect else va or vb
            if v != vp or ct == 0:
                cnts.append(cc)
                cc = 0

    return RLEs([RLE(h, w, len(cnts), torch.stack(cnts))])


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
    db = rleToBbox(dt, m)
    gb = rleToBbox(gt, n)
    o = bbIou(db, gb, m, n, iscrowd)
    for g in range(n):
        for d in range(m):
            if o[d, g] > 0:
                crowd = iscrowd is not None and iscrowd[g]
                if dt[d].h != gt[g].h or dt[d].w != gt[g].w:
                    o[d, g] = -1
                    continue
                ka = dt[d].m
                kb = gt[g].m
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
                    c = min(ca, cb).clone()
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
                o[d, g] = i / u
    return o


# TODO: Note used in python api
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
        dt: Detection bounding boxes (shape: [m, 4]).
        gt: Ground truth bounding boxes (shape: [n, 4]).
        m: Number of detection bounding boxes.
        n: Number of ground truth bounding boxes.
        iscrowd: List indicating if a ground truth bounding box is a crowd.

    Returns:
        IoU values for each detection and ground truth pair (shape: [m, n]).
    """
    # Convert the bounding boxes from [x1, y1, width, height] to [x1, y1, x2, y2]
    dt = torch.cat((dt[:, :2], dt[:, :2] + dt[:, 2:]), dim=1)
    gt = torch.cat((gt[:, :2], gt[:, :2] + gt[:, 2:]), dim=1)

    # Calculate area for detection and ground truth boxes
    dt_area = (dt[:, 2] - dt[:, 0]) * (dt[:, 3] - dt[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    # Compute intersection
    intersect_min = torch.max(dt[:, None, :2], gt[:, :2])  # [m, n, 2]
    intersect_max = torch.min(dt[:, None, 2:], gt[:, 2:])  # [m, n, 2]
    intersect_wh = torch.clamp(intersect_max - intersect_min, min=0)  # [m, n, 2]
    intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]  # [m, n]

    # Compute union
    union_area = dt_area[:, None] + gt_area - intersect_area
    union_area[torch.tensor(iscrowd)] = dt_area[:, None]  # Adjust for crowd

    # Compute IoU
    iou = intersect_area / union_area

    return iou


# TODO: Note used in python api
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
                    u = bbIou(dt[i], dt[j], 1, 1, None)
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
        h, _w, m = R[i].h, R[i].w, R[i].m
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


# TODO: fix input tensor form
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
    # Precompute the xy coordinates for all bounding boxes
    xs = bb[:, 0]
    ys = bb[:, 1]
    xe = xs + bb[:, 2]
    ye = ys + bb[:, 3]

    # Stack and reshape to get the xy tensor for all bounding boxes
    xy = torch.stack([xs, ys, xs, ye, xe, ye, xe, ys], dim=1).view(n, 8)

    # Apply rleFrPoly (assumed to be vectorized) to each bounding box
    r = [rleFrPoly(xy[i], 4, h, w) for i in range(n)]
    return r


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
    device = xy.device
    # upsample and get discrete points densely along entire boundary
    scale = 5.0
    x = ((scale * xy[0::2]) + 0.5).int()
    x = torch.cat((x, x[0:1]))
    y = ((scale * xy[1::2]) + 0.5).int()
    y = torch.cat((y, y[0:1]))

    max_diff = torch.maximum(torch.abs(torch.diff(x)), torch.abs(torch.diff(y))) + 1
    m = torch.sum(max_diff).int()

    xs = x[:-1]
    xe = x[1:]
    ys = y[:-1]
    ye = y[1:]
    dx = torch.abs(xe - xs)
    dy = torch.abs(ys - ye)
    flip = ((dx >= dy) & (xs > xe)) | ((dx < dy) & (ys > ye))
    _xs = torch.where(flip, xe, xs)
    _xe = torch.where(flip, xs, xe)
    _ys = torch.where(flip, ye, ys)
    _ye = torch.where(flip, ys, ye)
    xs = _xs
    xe = _xe
    ys = _ys
    ye = _ye
    xy_cond = dx >= dy
    s = torch.where(xy_cond, (ye - ys).to(dtype=torch.float64) / dx, (xe - xs).to(dtype=torch.float64) / dy)  # double
    seq_lens = torch.where(xy_cond, dx, dy).unsqueeze(-1)

    max_len = torch.max(seq_lens)
    d = torch.arange(0, max_len + 1, device=device).unsqueeze(0)
    d = d.expand(seq_lens.size(0), d.size(1))
    d_mask = d <= seq_lens

    dxy = torch.where(xy_cond, dx, dy)
    t = torch.where(flip.unsqueeze(1), dxy.unsqueeze(1) - d, d)
    # t = torch.clamp(t, 0, d)

    u = torch.where(xy_cond.unsqueeze(1), t + xs.unsqueeze(1), (xs.unsqueeze(1) + s.unsqueeze(1) * t + 0.5).int())
    v = torch.where(xy_cond.unsqueeze(1), (ys.unsqueeze(1) + s.unsqueeze(1) * t + 0.5).int(), t + ys.unsqueeze(1))

    u = u[d_mask]
    v = v[d_mask]
    # assert u.size(0) == m  # noqa: S101

    # get points along y-boundary and downsample
    changed = u[1:] != u[:-1]
    xd = torch.where(u[1:] < u[:-1], u[1:], u[1:] - 1)
    xd = (xd + 0.5) / scale - 0.5

    fl_mask = torch.floor(xd) != xd
    sm_mask = xd < 0
    lg_mask = xd > w - 1
    or_mask = torch.logical_or(fl_mask, torch.logical_or(sm_mask, lg_mask))
    cont_mask = torch.logical_not(or_mask)

    full_mask = torch.logical_and(cont_mask, changed)

    yd = torch.where(v[1:] < v[:-1], v[1:], v[:-1])
    yd = (yd + 0.5) / scale - 0.5
    yd = torch.clamp(yd, 0, h)
    yd = torch.ceil(yd)

    xd = xd[full_mask]
    yd = yd[full_mask]
    # compute rle encoding given y-boundary points
    a = xd * h + yd
    a = torch.cat((a, torch.tensor([h * w], device=device)))

    sorted, _ = torch.sort(a)

    a = torch.diff(sorted, prepend=torch.tensor([0], device=device)).to(dtype=torch.int)

    b = torch.zeros_like(a, device=device)
    j, m = 0, 0
    while j < len(a):
        if a[j] > 0:
            b[m] = a[j]
            m += 1
            j += 1
        else:
            j += 1
            if j < len(a):
                b[m - 1] += a[j]
                j += 1

    b = b[:m]

    # Initialize RLE with the counts
    r = RLE(h=h, w=w, m=len(b), cnts=b)
    return r


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
