import numpy as np
import pycocotools.mask as mask
import pytest
import pytorchcocotools.mask as tmask
import torch


def test_internal_encode():
    pt_mask = torch.zeros((25, 25), dtype=torch.uint8)
    pt_mask[:5, :5] = 1
    np_mask = np.asfortranarray(pt_mask.numpy())
    # compute the area
    rle1 = mask.encode(np_mask)
    rle2 = tmask.encode(pt_mask)

    import ctypes

    class MyStruct(ctypes.Structure):
        _fields_ = [
            ("h", ctypes.c_ulong),
            ("w", ctypes.c_ulong),
            ("m", ctypes.c_ulong),
            ("cnts", ctypes.POINTER(ctypes.c_uint)),
            # Add other fields here
        ]

    from pycocotools import _mask
    from pytorchcocotools import _mask as _tmask

    asd = _mask._frString([rle1])
    asd2 = _tmask._frString([rle2])

    my_struct = ctypes.cast(hex(id(asd)), ctypes.POINTER(MyStruct)).contents

    assert my_struct.h == 25
