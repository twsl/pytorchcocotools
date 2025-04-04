import numpy as np
import pycocotools.mask as mask_util_np
import pytest
import torch

from pytorchcocotools.internal.entities import RleObj
import pytorchcocotools.mask as mask_util_pt


def get_invalid_rle() -> RleObj:
    rle = RleObj(
        size=[1024, 1024],
        counts=b"jd`0=`o06J5L4M3L3N2N2N2N2N1O2N2N101N1O2O0O1O2N100O1O2N100O1O1O1O1O101N1O1O1O1O1O1O101N1O100O101O0O100000000000000001O00001O1O0O2O1N3N1N3N3L5Kh0XO6J4K5L5Id[o5N]dPJ7K4K4M3N2M3N2N1O2N100O2O0O1000O01000O101N1O1O2N2N2M3M3M4J7Inml5H[RSJ6L2N2N2N2O000000000000O2O1N2N2Mkm81SRG6L3L3N2O1N2N2O0O2O00001O0000000000O2O001N2O0O2N2N3M3L5JRjf6MPVYI8J4L3N3M2N1O2O1N101N1000000O10000001O000O101N101N1O2N2N2N3L4L7FWZ_50ne`J0000001O000000001O0000001O1O0N3M3N1O2N2N2O1N2O001N2`RO^O`k0c0[TOEak0;\\\\TOJbk07\\\\TOLck03[TO0dk01ZTO2dk0OYTO4gk0KXTO7gk0IXTO8ik0HUTO:kk0ETTO=lk0CRTO>Pl0@oSOb0Rl0\\\\OmSOe0Tl0[OjSOg0Ul0YOiSOi0Wl0XOgSOi0Yl0WOeSOk0[l0VOaSOn0kh0cNmYO",
    )
    return rle


def get_zero_leading_rle() -> RleObj:
    # A foreground segment of length 0 was not previously handled correctly.
    # This input rle has 3 leading zeros.
    rle = RleObj(
        size=[1350, 1080],
        counts=b"000lg0Zb01O00001O00001O001O00001O00001O001O00001O01O2N3M3M3M2N3M3N2M3M2N1O1O1O1O2N1O1O1O2N1O1O101N1O1O1O2N1O1O1O2N3M2N1O2N1O2O0O2N1O1O2N1O2N1O2N1O2N1O2N1O2O0O2N1O3M2N1O2N2N2N2N2N1O2N2N2N2N1O2N2N2N2N2N1N3N2N00O1O1O1O100000000000000O100000000000000001O0000001O00001O0O5L7I5K4L4L3M2N2N2N1O2m]OoXOm`0Sg0j^OVYOTa0lf0c^O]YO[a0ef0\\^OdYOba0bg0N2N2N2N2N2N2N2N2N2N2N2N2N2N2N2N2N3M2M4M2N3M2N3M2N3M2N3M2N3M2N3M2N3M2N3M2M4M2N2N3M2M4M2N2N3M2M3N3M2N3M2M3N3M2N2N3L3N2N3M2N3L3N2N3M5J4M3M4L3M3L5L3M3M4L3L4\\EXTOd6jo0K6J5K6I4M1O1O1O1N2O1O1O001N2O00001O0O101O000O2O00001N101N101N2N101N101N101N2O0O2O0O2O0O2O1N101N2N2O1N2O1N2O1N101N2O1N2O1N2O0O2O1N2N2O1N2O0O2O1N2O1N2N2N1N4M2N2M4M2N3L3N2N3L3N3L3N2N3L3N2N3L3M4L3M3M4L3M5K5K5K6J5K5K6J7I7I7Ibijn0",
    )
    return rle


@pytest.mark.parametrize("rle", [get_invalid_rle()])
def test_invalid_rle_counts_np(rle) -> None:
    with pytest.raises(ValueError) as error:
        mask_util_np.decode(rle)
    assert str(error.value) == "Invalid RLE mask representation"


@pytest.mark.parametrize("rle", [get_invalid_rle()])
def test_invalid_rle_counts_pt(rle) -> None:
    # pycocotools raises a ValueError, pytorchcocotools raises a runtime error
    with pytest.raises(RuntimeError) as error:
        mask_util_pt.decode(rle)
    assert str(error.value) == "upper bound and larger bound inconsistent with step sign"


@pytest.mark.parametrize("rle", [get_zero_leading_rle()])
def test_zero_leading_rle_np(rle) -> None:
    orig_bbox = mask_util_np.toBbox(rle)  # [  0.,   0., 331., 776.]
    mask = mask_util_np.decode(rle)
    rle_new = mask_util_np.encode(mask)
    new_bbox = mask_util_np.toBbox(rle_new)  # [  0.,   0., 331., 776.]
    assert np.equal(orig_bbox, new_bbox).all()


@pytest.mark.parametrize("rle", [get_zero_leading_rle()])
def test_zero_leading_rle_pt(rle) -> None:
    orig_bbox = mask_util_pt.toBbox(rle)  # [  0.,   0., 331., 776.]
    mask = mask_util_pt.decode(rle)
    rle_new = mask_util_pt.encode(mask)
    assert rle.counts[2:] == rle_new[0].counts
    new_bbox = mask_util_pt.toBbox(rle_new)  # [  0.,   0., 331., 776.]
    assert torch.equal(orig_bbox, new_bbox)
