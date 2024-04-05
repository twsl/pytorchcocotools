from pytorchcocotools._entities import (
    BB,
    RLE,
    Mask,
    RleObj,
    RleObjs,
    RLEs,
)
from pytorchcocotools._maskApi import (
    bbIou,
    rleArea,
    rleDecode,
    rleEncode,
    rleFrBbox,
    rleFrPoly,
    rleFrString,
    rleIou,
    rleMerge,
    rleToBbox,
    rleToString,
)
import torch
from torch import Tensor


def _toString(Rs: RLEs) -> RleObjs:  # noqa: N802, N803
    """Internal conversion from Python RLEs object to compressed RLE format.

    Args:
        Rs: _description_

    Returns:
        _description_
    """
    objs = [RleObj(size=[r.h, r.w], counts=rleToString(r)) for r in Rs]
    return RleObjs(objs)


def _frString(rleObjs: RleObjs) -> RLEs:  # noqa: N802, N803
    """Internal conversion from compressed RLE format to Python RLEs object.

    Args:
        rleObjs: List of rle encoded masks.

    Returns:
        _description_
    """
    rles = [
        rleFrString(
            str.encode(obj["counts"]) if isinstance(obj["counts"], str) else obj["counts"],
            obj["size"][0],
            obj["size"][1],
        )
        for obj in rleObjs
    ]
    return RLEs(rles)


def encode(mask: Mask) -> RleObjs:
    """Encode mask to RLEs objects, list of RLE string can be generated by RLEs member function.

    Args:
        mask: _description_

    Returns:
        _description_
    """
    # np.ndarray[np.uint8_t, ndim=3, mode='fortran']
    h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
    Rs = rleEncode(mask, h, w, n)  # noqa: N806
    objs = _toString(Rs)
    return objs


def decode(rleObjs: RleObjs) -> Mask:  # noqa: N803
    """Decode mask from compressed list of RLE string or RLEs object.

    Args:
        rleObjs: _description_

    Returns:
        _description_
    """
    rs = _frString(rleObjs)
    _h, _w, n = rs[0].h, rs[0].w, rs.n
    masks = rleDecode(rs, n)
    return masks


def merge(rleObjs: RleObjs, intersect: bool = False) -> RleObj:  # noqa: N803
    """Merges multiple rles into one rle mask by taking union (OR) or intersection (AND).

    Args:
        rleObjs: _description_
        intersect: _description_. Defaults to False.

    Returns:
        _description_
    """
    rs = _frString(rleObjs)
    r = rleMerge(rs, rs.n, intersect)
    obj = _toString(r)[0]
    return obj


def area(rleObjs: RleObjs) -> list[int]:  # noqa: N803
    rs = _frString(rleObjs)
    a = rleArea(rs, rs.n)
    return a


# iou computation. support function overload (RLEs-RLEs and bbox-bbox).
def iou(dt: RLEs | BB | list | Tensor, gt: RLEs | BB | list | Tensor, pyiscrowd: list[bool]) -> Tensor:
    def _preproc(objs):
        if len(objs) == 0:
            return objs
        if isinstance(objs, Tensor):
            if len(objs.shape) == 1:
                # TODO: figure out, why pycocotools didn't use the shape, propably just another error?
                # objs = objs.reshape((objs[0], 1))
                objs = objs.reshape((objs.shape[0], 1))
            # check if it's Nx4 bbox
            if not len(objs.shape) == 2 or not objs.shape[1] == 4:
                raise Exception("Tensor input is only for *bounding boxes* and should have Nx4 dimension")  # noqa: TRY002
            objs = objs.to(dtype=torch.float32)  # TODO: originally double is used, why???
        elif isinstance(objs, list):
            # check if list is in box format and convert it to torch.Tensor
            isbox = bool(torch.all(Tensor([(len(obj) == 4) and (isinstance(obj, list | Tensor)) for obj in objs])))
            isrle = bool(torch.all(Tensor([isinstance(obj, dict) for obj in objs])))
            if isbox:
                objs = torch.tensor(objs, dtype=torch.float32)
                if len(objs.shape) == 1:
                    objs = objs.reshape((1, objs.shape[0]))
            elif isrle:
                objs = _frString(objs)
            else:
                raise Exception("list input can be bounding box (Nx4) or RLEs ([RLE])")  # noqa: TRY002
        else:
            raise TypeError(
                "Unrecognized type.  The following type: RLEs (rle), torch.Tensor (box), and list (box) are supported."
            )
        return objs

    def _len(obj):
        if type(obj) == RLEs:
            return obj.n
        elif len(obj) == 0:
            return 0
        elif isinstance(obj, Tensor):
            return obj.shape[0]
        return 0

    is_crowd = pyiscrowd
    dt = _preproc(dt)
    gt = _preproc(gt)
    m = _len(dt)
    n = _len(gt)
    crowd_length = len(is_crowd)
    assert crowd_length == n, "iou(iscrowd=) must have the same length as gt"  # noqa: S101
    if m == 0 or n == 0:
        return Tensor([])  # TODO: fix return type to be consistent
    if not type(dt) == type(gt):
        raise Exception("The dt and gt should have the same data type, either RLEs, list or torch.Tensor")  # noqa: TRY002
    _iouFun = rleIou if isinstance(dt, RLEs) else bbIou if isinstance(dt, Tensor) else None  # noqa: N806
    if _iouFun is None:
        raise Exception("input data type not allowed.")  # noqa: TRY002

    iou = _iouFun(dt, gt, m, n, is_crowd)
    # return iou.reshape((m, n), order="F")
    return iou


def toBbox(rleObjs: RleObjs) -> BB:  # noqa: N803, N802
    rs = _frString(rleObjs)
    n = rs.n
    bb = rleToBbox(rs, n)
    return bb


def frBbox(bb: BB, h: int, w: int) -> RleObjs:  # noqa: N802
    n = bb.shape[0]
    rs = rleFrBbox(bb, h, w, n)
    objs = _toString(rs)
    return objs


def frPoly(poly: list[list[float]] | Tensor, h: int, w: int) -> RleObjs:  # noqa: N802
    rs = []  # RLEs(n)
    for p in poly:
        np_poly = p.to(dtype=torch.float64) if isinstance(p, Tensor) else torch.tensor(p, dtype=torch.float64)
        rs.append(rleFrPoly(np_poly, int(len(p) / 2), h, w))
    objs = _toString(RLEs(rs))
    return RleObjs(objs)


def frUncompressedRLE(ucRles: list[dict], h: int, w: int) -> RleObjs:  # noqa: N803, N802
    n = len(ucRles)
    objs = []
    for i in range(n):
        cnts = torch.tensor(ucRles[i]["counts"], dtype=torch.int)
        r = RLE(ucRles[i]["size"][0], ucRles[i]["size"][1], len(cnts), cnts)
        objs.append(_toString(RLEs([r]))[0])
    return RleObjs(objs)


def frPyObjects(pyobj: Tensor | list[list[int]] | list[dict] | dict, h: int, w: int) -> RleObjs | RleObj:  # noqa: N802
    # encode rle from a list of python objects
    if isinstance(pyobj, Tensor):
        return frBbox(pyobj, h, w)
    elif isinstance(pyobj, list) and len(pyobj[0]) == 4:  # not working in pycocotools
        return frBbox(Tensor(pyobj), h, w)
    elif isinstance(pyobj, list) and len(pyobj[0]) > 4:
        return frPoly(pyobj, h, w)
    elif isinstance(pyobj, list) and isinstance(pyobj[0], dict) and "counts" in pyobj[0] and "size" in pyobj[0]:
        return frUncompressedRLE(pyobj, h, w)
    # encode rle from single python object
    elif isinstance(pyobj, list) and len(pyobj) == 4:  # not working in pycocotools
        return frBbox(Tensor([pyobj]), h, w)[0]
    elif isinstance(pyobj, list) and len(pyobj) > 4:
        return frPoly([pyobj], h, w)[0]
    elif isinstance(pyobj, dict) and "counts" in pyobj and "size" in pyobj:
        return frUncompressedRLE([pyobj], h, w)[0]
    else:
        raise Exception("input type is not supported.")  # noqa: TRY002
