import numpy as np
import pycocotools.mask as mask


def test_area():
    # Test area() method
    data = np.asfortranarray(np.ones((10, 10), dtype=np.uint8))
    rle = mask.encode(data)
    assert mask.area(rle) == 100


def test_toBbox():  # noqa: N802
    # Test toBbox() method
    data = np.asfortranarray(np.ones((10, 10), dtype=np.uint8))
    rle = mask.encode(data)
    bbox = mask.toBbox(rle)
    assert np.array_equal(bbox, [0, 0, 10, 10])


def test_frPyObjects():  # noqa: N802
    # Test frPyObjects() method
    polygons = [[0, 0, 0, 10, 10, 10, 10, 0]]
    obj = mask.frPyObjects(polygons, 10, 10)
    data1 = np.asfortranarray(np.ones((10, 10), dtype=np.uint8))
    rle = mask.encode(data1)
    assert obj[0] == rle


def test_encode():
    # Test encode() method
    data1 = np.asfortranarray(np.ones((10, 10), dtype=np.uint8))
    rle = mask.encode(data1)
    assert rle["size"] == [10, 10]
    # assert len(rle['counts']) == 2


def test_decode():
    # Test decode() method
    data1 = np.asfortranarray(np.ones((10, 10), dtype=np.uint8))
    rle = mask.encode(data1)
    mask_arr = mask.decode(rle)
    assert np.array_equal(mask_arr, np.ones((10, 10), dtype=np.uint8))


def test_merge():
    # Test merge() method
    data1 = np.asfortranarray(np.ones((10, 10), dtype=np.uint8))
    data2 = np.asfortranarray(np.zeros((10, 10), dtype=np.uint8))
    data1[:, 0] = 0
    data2[:, 0] = 1
    rle1 = mask.encode(data1)
    rle2 = mask.encode(data2)
    merged_rle = mask.merge([rle1, rle2])
    mask_decoded = mask.decode(merged_rle)
    assert np.array_equal(mask_decoded, np.ones((10, 10), dtype=np.uint8))


def test_iou():
    # Test iou() method
    data1 = np.asfortranarray(np.ones((10, 10), dtype=np.uint8))
    data2 = np.asfortranarray(np.zeros((10, 10), dtype=np.uint8))
    rle1 = mask.encode(data1)
    rle2 = mask.encode(data2)
    iou = mask.iou([rle1], [rle2], [0])
    assert iou == [0.0]


def test_frPoly():  # noqa: N802
    # Test frPoly() method
    polygon = [[0, 0, 0, 10, 10, 10, 10, 0]]
    poly = mask._mask.frPoly(polygon, 10, 10)
    # poly_mask = mask.encode(poly)
    data1 = np.asfortranarray(np.ones((10, 10), dtype=np.uint8))
    rle1 = mask.encode(data1)

    assert poly[0] == rle1
