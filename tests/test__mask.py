import numpy as np
import pycocotools._mask as _mask
import pytest
import pytorchcocotools._mask as _tmask
import torch


# test the function frBbox
def test_frBbox():  # noqa: N802
    # create a bounding box
    bbox = [10, 10, 10, 10]
    # convert the bounding box to a mask
    mask1 = _mask.frBbox(np.array([bbox], dtype=np.double), 100, 100)
    mask2 = _tmask.frBbox(torch.Tensor(bbox), 100, 100)
    # compare the results
    assert mask1 == mask2
    assert mask1["counts"] == b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    assert mask1["size"] == [100, 100]


# test the function frPoly
def test_frPoly():  # noqa: N802
    # create a polygon
    poly = [[10, 10, 20, 10, 20, 20, 10, 20]]
    # convert the polygon to a mask
    mask1 = _mask.frPoly(poly, 100, 100)
    mask2 = _tmask.frPoly(poly, 100, 100)
    # compare the results
    assert mask1 == mask2
    assert mask1["counts"] == b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    assert mask1["size"] == [100, 100]


# test the function frUncompressedRLE
def test_frUncompressedRLE():  # noqa: N802
    # create a mask
    mask = np.zeros((100, 100, 1), dtype=np.uint8, order="F")
    mask[10:20, 10:20, :] = 1
    # encode the mask
    encoded = _mask.encode(mask)
    # convert the mask to a polygon
    poly1 = _mask.frUncompressedRLE(encoded, 100, 100)
    poly2 = _tmask.frUncompressedRLE(encoded, 100, 100)
    # compare the results
    assert poly1 == poly2
    assert poly1 == [[10, 10, 20, 10, 20, 20, 10, 20]]


def xtest_encode():
    # create a binary mask array
    mask_array = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    # encode the mask
    encoded_mask = coco_mask.encode(mask_array)

    # check that the encoded mask is a dictionary with the expected keys
    assert isinstance(encoded_mask, dict)
    assert set(encoded_mask.keys()) == {"size", "counts"}

    # check that the size of the mask is correct
    assert encoded_mask["size"] == [3, 3]

    # check that the count values are correct
    assert encoded_mask["counts"] == b"\x06\x0c\x06\x06\x06\x0c"


def xtest_decode():
    # create an encoded mask
    encoded_mask = {"size": [3, 3], "counts": b"\x06\x0c\x06\x06\x06\x0c"}

    # decode the mask
    mask_array = coco_mask.decode(encoded_mask)

    # check that the decoded mask is a numpy array with the expected shape and values
    assert isinstance(mask_array, np.ndarray)
    assert mask_array.shape == (3, 3)
    assert np.all(mask_array == np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))


def xtest_area():
    # create a binary mask array
    mask_array = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    # calculate the area of the mask
    area = coco_mask.area(mask_array)

    # check that the calculated area is correct
    assert area == 5


def xtest_toBbox():
    # create a binary mask array
    mask_array = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    # calculate the bounding box of the mask
    bbox = coco_mask.toBbox(mask_array)

    # check that the calculated bounding box is correct
    assert np.all(bbox == np.array([0, 1, 3, 3]))
