import numpy as np
from pycocotools import _EncodedRLE
import pycocotools.mask as mask_util_np
import pytest
import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.internal.entities import RleObj, RleObjs
import pytorchcocotools.mask as mask_util_pt


def get_full_img() -> Tensor:
    return torch.tensor([[0, 1], [1, 1]])


def get_non_full_img() -> Tensor:
    mask = torch.zeros((10, 10), dtype=torch.uint8)
    mask[2:4, 3:6] = 1
    return mask


def _encode_np(x: np.ndarray) -> _EncodedRLE:
    return mask_util_np.encode(np.asfortranarray(x, np.uint8))


def _encode_pt(x: tv.Mask) -> RleObj | RleObjs:
    return mask_util_pt.encode(x)


@pytest.mark.parametrize("img", [get_full_img()])
def test_tobbox_full_image_np(img: Tensor) -> None:
    mask = img.numpy()
    bbox = mask_util_np.toBbox(_encode_np(mask))
    assert (bbox == np.array([0, 0, 2, 2], dtype="float32")).all()


@pytest.mark.parametrize("img", [get_full_img()])
def test_tobbox_full_image_pt(img: Tensor) -> None:
    mask = tv.Mask(img)
    bbox = mask_util_pt.toBbox(_encode_pt(mask))
    assert torch.allclose(bbox, torch.tensor([0, 0, 2, 2], dtype=torch.int32))


@pytest.mark.parametrize("img", [get_non_full_img()])
def test_tobbox_non_full_image_np(img: Tensor) -> None:
    mask = img.numpy()
    bbox = mask_util_np.toBbox(_encode_np(mask))
    assert (bbox == np.array([3, 2, 3, 2], dtype="float32")).all()


# bugfix by piotr in ff4a47150bf66
@pytest.mark.parametrize("img", [get_non_full_img()])
def test_tobbox_non_full_image_pt(img: Tensor) -> None:
    mask = tv.Mask(img)
    bbox = mask_util_pt.toBbox(_encode_pt(mask))
    assert torch.allclose(bbox, torch.tensor([3, 2, 3, 2], dtype=torch.int32))
