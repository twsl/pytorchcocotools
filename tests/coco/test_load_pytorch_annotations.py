from pycocotools.coco import COCO as COCOnp  # noqa: N811
import pytest
from pytest_cases import parametrize_with_cases
import torch

from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


class LoadPyTorchAnnotationsCases:
    def case_test(self) -> tuple[torch.Tensor, list[dict]]:
        data = torch.Tensor([[397133, 48.51, 240.91, 247.03, 184.81, 0.999, 1]])
        result = [
            {
                "image_id": 397133,
                "bbox": [48.509998321533203125, 240.910003662109375, 247.029998779296875, 184.80999755859375],
                "score": 0.999000012874603271484375,
                "category_id": 1,
            }
        ]
        return (data, result)


@pytest.mark.benchmark(group="loadPyTorchAnnotations", warmup=True)
@parametrize_with_cases("data, result", cases=LoadPyTorchAnnotationsCases)
def test_loadPyTorchAnnotations_pt(benchmark, coco_pt: COCOpt, data: torch.Tensor, result: list[dict]) -> None:  # noqa: N802
    # get the category ids for the image with id
    ann_pt = benchmark(coco_pt.loadPyTorchAnnotations, data)
    # compare the results
    assert ann_pt == result


@pytest.mark.benchmark(group="loadPyTorchAnnotations", warmup=True)
@parametrize_with_cases("data, result", cases=LoadPyTorchAnnotationsCases)
def test_loadPyTorchAnnotations_np(benchmark, coco_np: COCOnp, data: torch.Tensor, result: list[dict]) -> None:  # noqa: N802
    # get the category ids for the image with id
    ann_np = benchmark(coco_np.loadNumpyAnnotations, data.numpy())
    # compare the results
    assert ann_np == result


@parametrize_with_cases("data, result", cases=LoadPyTorchAnnotationsCases)
def test_loadPyTorchAnnotations(coco_np: COCOnp, coco_pt: COCOpt, data: torch.Tensor, result: list[dict]) -> None:  # noqa: N802
    # test with a numpy array and a tensor [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    data_pt = data
    data_np = data.numpy()
    # get the annotations with id
    anns_np = coco_np.loadNumpyAnnotations(data_np)
    anns_pt = coco_pt.loadPyTorchAnnotations(data_pt)
    # compare the results
    for annnp, annpt in zip(anns_np, anns_pt, strict=False):
        for key in annnp:
            assert annnp[key] == annpt[key]
        assert annnp == annpt.__dict__
    for annnp, ann in zip(anns_np, result, strict=False):
        assert annnp == ann
