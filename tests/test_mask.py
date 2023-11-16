import numpy as np
import pycocotools.mask as mask
import pytest
import pytorchcocotools.mask as tmask
import torch


@pytest.fixture
def r() -> tmask.RleObj:
    return tmask.RleObj((100, 100), bytes(""))


@pytest.mark.parametrize(
    "min, max, h, w, area",
    [
        (0, 5, 25, 25, 25),
        (5, 10, 25, 25, 25),
    ],
)
def test_area(min: int, max: int, h: int, w: int, area: int):
    # create a mask
    pt_mask = torch.zeros((h, w), dtype=torch.uint8)
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
    "min, max, h, w, box",
    [
        (0, 10, 25, 25, [0, 0, 10, 10]),
        (0, 25, 25, 25, [0, 0, 25, 25]),
        (10, 20, 25, 25, [10, 10, 10, 10]),
    ],
)
def test_toBbox(min: int, max: int, h: int, w: int, box: list[int]):  # noqa: N802
    # create a mask
    pt_mask = torch.zeros((h, w), dtype=torch.uint8)
    pt_mask[min:max, min:max] = 1
    np_mask = np.asfortranarray(pt_mask.numpy())
    # compute the bounding box
    rle1 = mask.encode(np_mask)
    rle2 = tmask.encode(pt_mask)
    bbox1 = mask.toBbox(rle1)
    bbox2 = tmask.toBbox(rle2)
    # compare the results
    assert np.all(bbox1 == bbox2.numpy())  # np.allclose(bbox1, bbox2.numpy())
    assert list(bbox2.numpy()) == box


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
def test_frPyObjects(poly: list[int], length: int, r: tmask.RleObj):  # noqa: N802
    # convert the polygon to a mask
    mask1 = mask.frPyObjects(poly, length, length)
    mask2 = tmask.frPyObjects(poly, length, length)

    # compare the results
    assert mask2[0]["counts"] == mask1[0]["counts"]
    assert mask2[0]["size"] == mask1[0]["size"]


@pytest.mark.parametrize(
    "min, max, h, w",
    [
        (0, 10, 100, 100),
        (10, 20, 100, 100),
        (0, 100, 100, 100),
    ],
)
def test_decode(min: int, max: int, h: int, w: int):
    # create a mask
    pt_mask = torch.zeros((h, w, 2), dtype=torch.uint8)
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
    "min, max, h, w",
    [
        (0, 10, 100, 100),
        (10, 20, 100, 100),
        (0, 100, 100, 100),
    ],
)
def test_encode(min: int, max: int, h: int, w: int):
    # create a mask
    pt_mask = torch.zeros((h, w, 2), dtype=torch.uint8)
    pt_mask[min:max, min:max, :] = 1
    np_mask = np.asfortranarray(pt_mask.numpy())
    # encode the mask
    encoded1 = mask.encode(np_mask)
    encoded2 = tmask.encode(pt_mask)
    # compare the results
    assert encoded1[0]["counts"] == encoded2[0]["counts"]
    assert encoded1[0]["size"] == encoded2[0]["size"]


@pytest.mark.parametrize(
    "min1, max1, min2, max2, h, w",
    [
        (10, 20, 15, 25, 100, 100),
    ],
)
def test_merge(min1: int, max1: int, min2: int, max2: int, h: int, w: int):
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
    # merge the masks
    merged1 = mask.merge([np_rle1, np_rle2], intersect=False)
    merged2 = tmask.merge([pt_rle1, pt_rle2], intersect=False)
    # compare the results
    assert merged1 == merged2
    assert merged2["counts"] == merged1["counts"]
    assert merged2["size"] == merged1["size"]
