import numpy as np
import pycocotools.mask as mask
import pytest
from pytest_cases import case, parametrize_with_cases
import pytorchcocotools.mask as tmask
import torch
from torch import Tensor


class PyObjectsCases:
    h = 25
    w = 25

    def case_bbox_tensor(self):
        return (
            self.h,
            self.w,
            torch.tensor([[10, 10, 10, 10]], dtype=torch.float64),
            [{"size": [self.h, self.w], "counts": b"T8:?00000000000000000c3"}],
        )

    @pytest.mark.skip(reason="Original pycocotools implementation is wrong")
    def case_bbox_list(self):
        return (
            self.h,
            self.w,
            [
                [10, 10, 10, 10],
            ],
            [{"size": [self.h, self.w], "counts": b"T8:?00000000000000000c3"}],
        )

    @pytest.mark.skip(reason="Original pycocotools implementation is wrong")
    def case_bbox(self):
        return (
            self.h,
            self.w,
            [10, 10, 10, 10],
            {"size": [self.h, self.w], "counts": b"T8:?00000000000000000c3"},
        )

    def case_poly_list(self):
        return (
            self.h,
            self.w,
            [[10, 10, 20, 10, 20, 20, 21, 21, 10, 20]],
            [{"size": [self.h, self.w], "counts": b"T8:?00000000001O00000:F`2"}],
        )

    @pytest.mark.skip(reason="Original pycocotools implementation is wrong")
    def case_poly(self):
        return (
            self.h,
            self.w,
            [10, 10, 20, 10, 20, 20, 21, 21, 10, 20],
            {"size": [self.h, self.w], "counts": b"T8:?00000000001O00000:F`2"},
        )

    def case_uncompr_list(self):
        return (
            self.h,
            self.w,
            [{"size": [self.h, self.w], "counts": [130, 5, 20, 5, 20, 5, 20, 5, 20, 5, 390]}],
            [{"size": [self.h, self.w], "counts": b"R45d00000000b;"}],
        )

    def case_uncompr(self):
        return (
            self.h,
            self.w,
            {"size": [self.h, self.w], "counts": [130, 5, 20, 5, 20, 5, 20, 5, 20, 5, 390]},
            {"size": [self.h, self.w], "counts": b"R45d00000000b;"},
        )


@pytest.mark.benchmark(group="encode", warmup=True)
@parametrize_with_cases("h, w, obj, result", cases=PyObjectsCases)
def test_frPyObjects_pt(benchmark, h: int, w: int, obj: list[int] | list[list[int]] | list[dict] | dict, result):  # noqa: N802
    # convert the polygon to a mask
    # mask_pt = benchmark(tmask.frPyObjects, obj, h, w)
    mask_pt = tmask.frPyObjects(obj, h, w)

    # fix output
    if not isinstance(mask_pt, list):
        mask_pt = [mask_pt]
        result = [result]

    # compare the results
    assert mask_pt[0]["counts"] == result[0]["counts"]
    assert mask_pt[0]["size"] == result[0]["size"]


@pytest.mark.benchmark(group="pyObjects", warmup=True)
@parametrize_with_cases("h, w, obj, result", cases=PyObjectsCases)
def test_frPyObjects_np(benchmark, h: int, w: int, obj: list[int] | list[list[int]] | list[dict] | dict, result):  # noqa: N802
    # fix input
    if isinstance(obj, list):
        obj = [o.numpy() if isinstance(o, Tensor) else o for o in obj]
    elif isinstance(obj, Tensor):
        obj = obj.numpy()

    # convert the polygon to a mask
    mask_np = benchmark(mask.frPyObjects, obj, h, w)

    # fix output
    if not isinstance(mask_np, list):
        mask_np = [mask_np]
        result = [result]

    # compare the results
    assert mask_np[0]["counts"] == result[0]["counts"]
    assert mask_np[0]["size"] == result[0]["size"]


@parametrize_with_cases("h, w, obj, result", cases=PyObjectsCases)
def test_frPyObjects(h: int, w: int, obj: list[int] | list[list[int]] | list[dict] | dict, result):  # noqa: N802
    if isinstance(obj, list):
        obj_np = [o.numpy() if isinstance(o, Tensor) else o for o in obj]
    elif isinstance(obj, Tensor):
        obj_np = obj.numpy()
    else:
        obj_np = obj

    # convert the polygon to a mask
    mask_np = mask.frPyObjects(obj_np, h, w)
    mask_pt = tmask.frPyObjects(obj, h, w)

    # fix output
    if not isinstance(mask_np, list) and not isinstance(mask_pt, list):
        mask_np = [mask_np]
        mask_pt = [mask_pt]
        result = [result]

    # compare the results
    assert mask_pt[0]["counts"] == mask_np[0]["counts"]
    assert mask_pt[0]["size"] == mask_np[0]["size"]
