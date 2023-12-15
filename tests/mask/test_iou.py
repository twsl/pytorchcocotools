import numpy as np
import pycocotools.mask as mask
import pytest
import pytorchcocotools.mask as tmask
import torch


@pytest.mark.parametrize(
    "min1, max1, min2, max2, h, w, iou",
    [
        (10, 20, 15, 25, 100, 100, 0.25),
    ],
)
def test_iou(min1: int, max1: int, min2: int, max2: int, h: int, w: int, iou: float):
    # create two masks
    pt_mask1 = torch.zeros((h, w), dtype=torch.uint8)
    pt_mask2 = torch.zeros((h, w), dtype=torch.uint8)
    pt_mask1[min1:max1, min1:max1] = 1
    pt_mask2[min2:max2, min2:max2] = 1
    np_mask1 = np.asfortranarray(pt_mask1.numpy())
    np_mask2 = np.asfortranarray(pt_mask2.numpy())
    # compute the iou
    np_rle1 = mask.encode(np_mask1)
    np_rle2 = mask.encode(np_mask2)
    pt_rle1 = tmask.encode(pt_mask1)
    pt_rle2 = tmask.encode(pt_mask2)
    iou1 = mask.iou(np_rle1, np_rle2, [0])
    iou2 = tmask.iou(pt_rle1, pt_rle2, [0])
    # compare the results
    assert iou1 == iou2
    assert iou1 == iou
