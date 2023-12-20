import numpy as np
from pycocotools.coco import COCO as COCO
import pytest
from pytorchcocotools.coco import tCOCO
import torch

# write unit tests after loading the file from data/examples.json to test the coco class


@pytest.fixture
def path() -> str:
    path = "../data/example.json"
    return path


@pytest.fixture
def coco1(path: str) -> COCO:
    return COCO(path)


@pytest.fixture
def coco2(path: str) -> tCOCO:
    return tCOCO(path)


def test_COCO_load(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    assert coco1 is not None
    assert coco2 is not None


# test the function getAnnIds for the coco class and the _coco class and compare results
def test_getAnnIds(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    # get the annotation ids for the image with id 397133
    annIds1 = coco1.getAnnIds(imgIds=397133)
    annIds2 = coco2.getAnnIds(imgIds=397133)
    # compare the results
    assert annIds1 == annIds2
    assert annIds1 == [2096753]


# test the function getCatIds for the coco class and the _coco class and compare results
def test_getCatIds(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    # get the category ids for the image with id 397133
    catIds1 = coco1.getCatIds(catIds=1)
    catIds2 = coco2.getCatIds(catIds=1)
    # compare the results
    assert catIds1 == catIds2
    assert catIds1 == [1]


# test the function getImgIds for the coco class and the _coco class and compare results
def test_getImgIds(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    # get the image ids for the category with id 1
    imgIds1 = coco1.getImgIds(catIds=1)
    imgIds2 = coco2.getImgIds(catIds=1)
    # compare the results
    assert imgIds1 == imgIds2
    assert imgIds1 == [397133]


# test the function loadAnns for the coco class and the _coco class and compare results
def test_loadAnns(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    # get the annotations with id 2096753
    ann1 = coco1.loadAnns(2096753)
    ann2 = coco2.loadAnns(2096753)
    # compare the results
    assert ann1 == ann2


# test the function loadCats for the coco class and the _coco class and compare results
def test_loadCats(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    # get the category with id 1
    cat1 = coco1.loadCats(1)
    cat2 = coco2.loadCats(1)
    # compare the results
    assert cat1 == cat2
    assert cat1 == [{"supercategory": "person", "id": 1, "name": "person"}]


# test the function loadImgs for the coco class and the _coco class and compare results
def test_loadImgs(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    # get the image with id 397133
    img1 = coco1.loadImgs(397133)
    img2 = coco2.loadImgs(397133)
    # compare the results
    assert img1 == img2
    assert img1 == [
        {
            "license": 1,
            "file_name": "000000397133.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
            "height": 427,
            "width": 640,
            "date_captured": "2013-11-14 17:02:52",
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
            "id": 397133,
        }
    ]


# test the function loadPyTorchAnnotations for the coco class and compare with the function loadNumpyAnnotations for the _coco class
def test_loadPyTorchAnnotations(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    # test with a numpy array and a tensor [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    data2 = torch.Tensor([[397133, 48.51, 240.91, 247.03, 184.81, 0.999, 1]])
    data1 = data2.numpy()

    # get the annotations with id 2096753
    ann1 = coco1.loadNumpyAnnotations(data1)
    ann2 = coco2.loadPyTorchAnnotations(data2)
    # compare the results
    assert ann1 == ann2


# test the function annToRLE for the coco class and the _coco class and compare results
def test_annToRLE(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    # test with an annotation dict object
    ann1 = coco1.loadAnns(2096753)
    ann2 = coco2.loadAnns(2096753)
    # get the RLE for the annotation
    rle_np = coco1.annToRLE(ann1[0])
    rle_pt = coco2.annToRLE(ann2[0])
    # compare the results
    assert rle_np == rle_pt
    assert rle_np["counts"] == rle_pt["counts"]
    assert rle_np["size"] == rle_pt["size"]


# @pytest.mark.skip(reason="Way too slow")
# test the function annToMask for the coco class and the _coco class and compare results
def test_annToMask(coco1: COCO, coco2: tCOCO) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_np = coco1.loadAnns(2096753)
    ann_pt = coco2.loadAnns(2096753)
    # get the mask for the annotation
    mask_np = coco1.annToMask(ann_np[0])
    mask_pt = coco2.annToMask(ann_pt[0])
    # compare the results
    # np.nonzero(mask_np)
    assert np.array_equal(mask_np, mask_pt.numpy())
