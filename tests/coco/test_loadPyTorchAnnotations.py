import numpy as np
from pycocotools.coco import COCO as COCO
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811  # noqa: N811
import torch


class LoadPyTorchAnnotationsCases:
    def case_test(self) -> tuple:
        data = torch.Tensor([[397133, 48.51, 240.91, 247.03, 184.81, 0.999, 1]])
        result = None
        return (data, result)


@pytest.mark.benchmark(group="loadPyTorchAnnotations", warmup=True)
@parametrize_with_cases("data, result", cases=LoadPyTorchAnnotationsCases)
def test_loadPyTorchAnnotations_pt(benchmark, coco_pt: COCOpt, data, result) -> None:  # noqa: N802
    # get the category ids for the image with id
    ann_pt = benchmark(coco_pt.loadPyTorchAnnotations(data))
    # compare the results
    assert ann_pt == result


@pytest.mark.benchmark(group="loadPyTorchAnnotations", warmup=True)
@parametrize_with_cases("data, result", cases=LoadPyTorchAnnotationsCases)
def test_loadPyTorchAnnotations_np(benchmark, coco_np: COCO, data, result) -> None:  # noqa: N802
    # get the category ids for the image with id
    ann_np = benchmark(coco_np.loadNumpyAnnotations(data.numpy()))
    # compare the results
    assert ann_np == result


@parametrize_with_cases("coco1, coco2, data, result", cases=LoadPyTorchAnnotationsCases)
def test_loadPyTorchAnnotations(coco1: COCO, coco2: COCOpt, data, result) -> None:  # noqa: N802
    # test with a numpy array and a tensor [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    data_pt = data
    data_np = data.numpy()
    # get the annotations with id
    ann_np = coco1.loadNumpyAnnotations(data_np)
    ann_pt = coco2.loadPyTorchAnnotations(data_pt)
    # compare the results
    assert ann_np == ann_pt
    assert ann_np == result
