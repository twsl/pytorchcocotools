class RLE:
    def __init__(self, h=0, w=0, m=0, cnts=None):
        self.h = h
        self.w = w
        self.m = m
        self.cnts = cnts if cnts is not None else [0] * m


def rle_encode(M, h, w, n):
    R = []
    a = w * h
    for i in range(n):
        T = M[i]
        cnts = []
        p = T[0]
        c = 0
        for j in T:
            if j == p:
                c += 1
            else:
                cnts.append(c)
                p = j
                c = 1
        cnts.append(c)  # Append the last run
        R.append(RLE(h, w, len(cnts), cnts))
    return R


def rle_decode(R, n, h, w):
    M = []
    for i in range(n):
        mask = []
        v = 0
        for cnt in R[i].cnts:
            mask.extend([v] * cnt)
            v = 1 - v  # Flip between 0 and 1
        M.append(mask[: h * w])  # Ensure the mask is the expected size
    return M


def rle_merge(R, intersect):
    if not R:
        return RLE()
    if len(R) == 1:
        return RLE(R[0].h, R[0].w, R[0].m, R[0].cnts.copy())

    h, w, m = R[0].h, R[0].w, R[0].m
    cnts = R[0].cnts.copy()
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


def rle_area(R, n):
    a = [0] * n
    for i in range(n):
        for j in range(1, R[i].m, 2):
            a[i] += R[i].cnts[j]
    return a


def rle_iou(dt, gt, m, n, iscrowd, o):
    db = [None] * (m * 4)
    gb = [None] * (n * 4)
    # Assuming rle_to_bbox and bb_iou are previously defined functions
    rle_to_bbox(dt, db, m)
    rle_to_bbox(gt, gb, n)
    bb_iou(db, gb, m, n, iscrowd, o)
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
                    u = rle_area(dt[d], 1)[0]
                o[g * m + d] = i / u


def rle_nms(dt, n, keep, thr):
    for i in range(n):
        keep[i] = 1
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j]:
                    u = [0]
                    rle_iou([dt[i]], [dt[j]], 1, 1, None, u)
                    if u[0] > thr:
                        keep[j] = 0


def bb_iou(dt, gt, m, n, iscrowd, o):
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


def bb_nms(dt, n, keep, thr):
    for i in range(n):
        keep[i] = 1
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j]:
                    u = [0]
                    bb_iou(dt[i * 4 : (i + 1) * 4], dt[j * 4 : (j + 1) * 4], 1, 1, None, u)
                    if u[0] > thr:
                        keep[j] = 0


def rle_to_bbox(R, n):
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


def rle_fr_bbox(bb, h, w, n):
    R = [None] * n
    for i in range(n):
        xs, xe = bb[4 * i], bb[4 * i] + bb[4 * i + 2]
        ys, ye = bb[4 * i + 1], bb[4 * i + 1] + bb[4 * i + 3]
        xy = [xs, ys, xs, ye, xe, ye, xe, ys]
        R[i] = rle_fr_poly(xy, 4, h, w)
    return R


def uint_compare(a, b):
    return (a > b) - (a < b)


def rle_fr_poly(xy, k, h, w):
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


def rle_to_string(R):
    """Convert RLE counts to a string using a similar method to LEB128 but using 6 bits per char."""
    s = []
    for i in range(R.m):
        x = R.cnts[i]
        if i > 2:
            x -= R.cnts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (c & 0x10) != 0 if x != -1 else x != 0
            if more:
                c |= 0x20
            c += 48
            s.append(chr(c))
    return "".join(s)


def rle_from_string(R, s, h, w):
    """Convert a string back to RLE counts."""
    m = 0
    p = 0
    cnts = []
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = (c & 0x20) != 0
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    R.__init__(h, w, m, cnts)
