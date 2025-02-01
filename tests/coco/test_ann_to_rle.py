from typing import cast

from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811
from pytorchcocotools.internal.entities import RleObj, RleObjs


class AnnToRLECases:
    def case_test(self) -> tuple[int, RleObj]:
        result = RleObj(
            size=[427, 640],
            counts=b"RT_32X<9SD3f;3ZDNb;5_DKU;DeDd05HU;b0kD_OS;b0nD]OQ;e0nD[OR;e0nD[OR;f0nDZOQ;f0oDZOQ;f0oDZOQ;g0oDXOQ;h0oDXOQ;h0oDXOQ;i0oDVOQ;j0oDVOQ;k0nDTOS;l0mDTOS;l0nDSOR;l0oDmNX;n0f0J5J6A^C0g<N=O001O0O2Omk^4",
        )
        return (2096753, result)


@pytest.mark.benchmark(group="annToRLE", warmup=True)
@parametrize_with_cases("ann_id, result", cases=AnnToRLECases)
def test_annToRLE_pt(benchmark: BenchmarkFixture, coco_pt: COCOpt, ann_id: int, result: RleObj) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_pt = coco_pt.loadAnns(ann_id)
    # get the mask for the annotation
    rle_pt = cast(RleObj, benchmark(coco_pt.annToRLE, ann_pt[0]))
    # compare the results
    assert rle_pt["counts"] == result["counts"]
    assert rle_pt["size"] == result["size"]


@pytest.mark.benchmark(group="annToRLE", warmup=True)
@parametrize_with_cases("ann_id, result", cases=AnnToRLECases)
def test_annToRLE_np(benchmark: BenchmarkFixture, coco_np: COCOnp, ann_id: int, result: RleObj) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_np = coco_np.loadAnns(ann_id)
    # get the mask for the annotation
    rle_np = cast(dict, benchmark(coco_np.annToRLE, ann_np[0]))
    # compare the results
    assert rle_np["counts"] == result["counts"]
    assert rle_np["size"] == result["size"]


@parametrize_with_cases("ann_id, result", cases=AnnToRLECases)
def test_annToRLE(coco_np: COCOnp, coco_pt: COCOpt, ann_id: int, result: RleObj) -> None:  # noqa: N802
    # test with an annotation dict object
    ann_np = coco_np.loadAnns(ann_id)
    ann_pt = coco_pt.loadAnns(ann_id)
    # get the RLE for the annotation
    rle_np: dict = coco_np.annToRLE(ann_np[0])  # pyright:ignore[reportCallIssue, reportArgumentType]
    rle_pt = cast(RleObj, coco_pt.annToRLE(ann_pt[0]))
    # compare the results
    assert rle_np == rle_pt.__dict__
    assert rle_np["counts"] == rle_pt["counts"]
    assert rle_np["size"] == rle_pt["size"]
    assert rle_np["counts"] == result["counts"]
    assert rle_np["size"] == result["size"]
