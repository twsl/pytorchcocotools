import numpy as np
import pycocotools.mask as mask
import pytest
import pytorchcocotools.mask as tmask
import torch


@pytest.mark.parametrize(
    "obj, h, w",
    [
        ([[10, 10, 10, 10]], 100, 100),  # bboxs
        ([[10, 10, 20, 10, 20, 20, 10, 20]], 100, 100),  # polys
        ([[10, 10, 20, 10, 20, 20, 21, 21, 10, 20]], 100, 100),  # polys
        ([10, 10, 10, 10], 100, 100),  # bbox
        # ([10, 10, 20, 10, 20, 20, 10, 20], 100, 100),  # poly
        # ([10, 10, 20, 10, 20, 20, 21, 21, 10, 20], 100, 100),  # poly
    ],
)
def test_frPyObjects(obj: list[int] | list[list[int]] | list[dict] | dict, h: int, w: int):  # noqa: N802
    # convert the polygon to a mask
    mask1 = mask.frPyObjects(obj, h, w)
    mask2 = tmask.frPyObjects(obj, h, w)

    # compare the results
    assert mask2[0]["counts"] == mask1[0]["counts"]
    assert mask2[0]["size"] == mask1[0]["size"]
