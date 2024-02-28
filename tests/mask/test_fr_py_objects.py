import pycocotools.mask as mask
import pytest
from pytest_cases import parametrize_with_cases
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

    def case_complex(self):
        return (
            427,
            640,
            [
                [
                    266.83,
                    189.37,
                    267.79,
                    175.29,
                    269.46,
                    170.04,
                    271.37,
                    165.98,
                    270.89,
                    163.12,
                    269.12,
                    159.54,
                    272.8,
                    156.44,
                    287.36,
                    156.44,
                    293.33,
                    157.87,
                    296.91,
                    160.49,
                    296.91,
                    161.21,
                    291.89,
                    161.92,
                    289.98,
                    165.03,
                    291.42,
                    169.56,
                    285.16,
                    196.54,
                ],
                [
                    266.35,
                    214.44,
                    270.41,
                    217.3,
                    276.38,
                    218.97,
                    282.11,
                    218.97,
                    285.93,
                    217.3,
                    286.88,
                    207.28,
                    267.07,
                    201.07,
                ],
            ],
            [
                {
                    "size": [427, 640],
                    "counts": b"\\`_3;j<6B@nCc0Q<@kCc0S<;01N10001O001O00001O001O0000O1L4K6K4L4B]COh<O<O001O0O2Omk^4",
                },
                {"size": [427, 640], "counts": b"RT_32n<<O100O0010O000010O0001O00001O000O101O0ISPc4"},
            ],
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
    for i in range(len(mask_np)):
        assert mask_pt[i]["counts"] == mask_np[i]["counts"]
        assert mask_pt[i]["size"] == mask_np[i]["size"]
        assert mask_np[i]["counts"] == result[i]["counts"]
        assert mask_np[i]["size"] == result[i]["size"]
