import numpy as np
import pycocotools.mask as mask
import pytest
import pytorchcocotools.mask as tmask
import torch


@pytest.fixture
def r() -> tmask.RleObj:
    pass


@pytest.mark.parametrize(
    "min, max, area",
    [
        (0, 5, 25),
        (0, 10, 100),
    ],
)
def test_area(min, max, area):
    # create a mask
    pt_mask = torch.zeros((25, 25), dtype=torch.uint8)
    pt_mask[min:max, min:max] = 1
    np_mask = np.asfortranarray(pt_mask.numpy())
    # compute the area
    rle1 = mask.encode(np_mask)
    rle2 = tmask.encode(pt_mask)
    area1 = mask.area(rle1)
    area2 = tmask.area(rle2)
    # compare the results
    assert area1 == area2
    assert area1 == area


@pytest.mark.parametrize(
    "min, max, box",
    [
        (10, 20, [10, 10, 10, 10]),
    ],
)
def test_toBbox(min, max, box):
    # create a mask
    pt_mask = torch.zeros((25, 25), dtype=torch.uint8)
    pt_mask[min:max, min:max] = 1
    np_mask = np.asfortranarray(pt_mask.numpy())
    # compute the bounding box
    rle1 = mask.encode(np_mask)
    rle2 = tmask.encode(pt_mask)
    bbox1 = mask.toBbox(rle1)
    bbox2 = tmask.toBbox(rle2)
    # compare the results
    assert bbox1 == bbox2
    assert bbox1 == box


@pytest.mark.parametrize(
    "min1, max1, min2, max2, iou",
    [
        (10, 20, 15, 25, 0.25),
    ],
)
def test_iou(min1, max1, min2, max2, iou):
    # create two masks
    pt_mask1 = torch.zeros((100, 100), dtype=torch.uint8)
    pt_mask2 = torch.zeros((100, 100), dtype=torch.uint8)
    pt_mask1[min:max, min:max] = 1
    pt_mask2[min:max, min:max] = 1
    np_mask1 = np.asfortranarray(pt_mask1.numpy())
    np_mask2 = np.asfortranarray(pt_mask2.numpy())
    # compute the iou
    np_rle1 = mask.encode(np_mask1)
    np_rle2 = mask.encode(np_mask2)
    pt_rle1 = mask.encode(pt_mask1)
    pt_rle2 = mask.encode(pt_mask2)
    iou1 = mask.iou([np_rle1, np_rle2], [0, 0])
    iou2 = tmask.iou([pt_rle1, pt_rle2], [0, 0])
    # compare the results
    assert iou1 == iou2
    assert iou1 == iou


@pytest.mark.parametrize(
    "poly, length, r",
    [
        ([[10, 10, 20, 10, 20, 20, 10, 20]], 100, r),
        ([[10, 10, 20, 10, 20, 20, 21, 21, 10, 20]], 100, r),
    ],
)
def test_frPyObjects(poly, length, r):
    # convert the polygon to a mask
    mask1 = mask.frPyObjects(poly, length, length)
    mask2 = tmask.frPyObjects(poly, length, length)

    data1 = np.asfortranarray(np.ones((length, length), dtype=np.uint8))
    mask.encode(data1)

    # compare the results
    assert mask1 == mask2
    assert mask2["counts"] == r["counts"]
    assert mask2["size"] == r["size"]


@pytest.mark.parametrize(
    "min, max, length, r",
    [
        (10, 20, 100, r),
    ],
)
def test_decode(min, max, length, r):
    # create a mask
    pt_mask = torch.zeros((length, length, 2), dtype=torch.uint8)
    pt_mask[min:max, min:max, :] = 1
    np_mask = np.asfortranarray(pt_mask.numpy())
    # encode the mask
    encoded = mask.encode(np_mask)
    # decode the mask
    decoded1 = mask.decode(encoded)
    decoded2 = tmask.decode(encoded)
    # compare the results
    assert np.array_equal(decoded1, np_mask)
    assert torch.equal(decoded2, pt_mask)
    assert np.array_equal(decoded1, decoded2.numpy())


@pytest.mark.parametrize(
    "min, max, length, r",
    [
        (10, 20, 100, r),
    ],
)
def test_encode(min, max, length, r):
    # create a mask
    pt_mask = torch.zeros((length, length, 2), dtype=torch.uint8)
    pt_mask[min:max, min:max, :] = 1
    np_mask = np.asfortranarray(pt_mask.numpy())
    # encode the mask
    encoded1 = mask.encode(np_mask)
    encoded2 = tmask.encode(pt_mask)
    # compare the results
    assert encoded1 == encoded2
    assert encoded2["counts"] == r["counts"]
    assert encoded2["size"] == r["size"]


@pytest.mark.parametrize(
    "min1, max1, min2, max2, length, r",
    [
        (10, 20, 15, 25, 100, r),
    ],
)
def test_merge(min1, max1, min2, max2, length, r):
    # create two masks
    pt_mask1 = torch.zeros((100, 100), dtype=torch.uint8)
    pt_mask2 = torch.zeros((100, 100), dtype=torch.uint8)
    pt_mask1[10:20, 10:20] = 1
    pt_mask2[15:25, 15:25] = 1
    np_mask1 = np.asfortranarray(pt_mask1.numpy())
    np_mask2 = np.asfortranarray(pt_mask2.numpy())
    # compute the iou
    np_rle1 = mask.encode(np_mask1)
    np_rle2 = mask.encode(np_mask2)
    pt_rle1 = mask.encode(pt_mask1)
    pt_rle2 = mask.encode(pt_mask2)
    # merge the masks
    merged1 = mask.merge([np_rle1, np_rle2], intersect=False)
    merged2 = tmask.merge([pt_rle1, pt_rle2], intersect=False)
    # compare the results
    assert merged1 == merged2
    assert merged2["counts"] == r["counts"]
    assert merged2["size"] == r["size"]
