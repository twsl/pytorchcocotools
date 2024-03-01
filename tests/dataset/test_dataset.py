import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.dataset import CocoDetection as CocoDetectionpt
from torchvision.datasets import CocoDetection as CocoDetectionnp


class DatasetCases:
    def case_example(self) -> tuple:
        return ("./data/coco", "./data/example.json", None)


@pytest.mark.benchmark(group="dataset", warmup=True)
@parametrize_with_cases("root, annotation_file, result", cases=DatasetCases)
def test_dataset_np(benchmark, root: str, annotation_file: str, result: int):
    benchmark.weave(CocoDetectionpt.__getitem__, lazy=True)
    dataset = CocoDetectionnp(root=root, annFile=annotation_file)
    item_np = dataset[0]
    # compare the results
    assert item_np == result


@pytest.mark.benchmark(group="dataset", warmup=True)
@parametrize_with_cases("root, annotation_file, result", cases=DatasetCases)
def test_dataset_pt(benchmark, root: str, annotation_file: str, result: int):
    benchmark.weave(CocoDetectionpt.__getitem__, lazy=True)
    dataset = CocoDetectionpt(root=root, annFile=annotation_file)
    item_pt = dataset[0]
    # compare the results
    assert item_pt == result


@parametrize_with_cases("root, annotation_file, result", cases=DatasetCases)
def test_dataset(root: str, annotation_file: str, result: int):
    dataset_np = CocoDetectionnp(root=root, annFile=annotation_file)
    dataset_pt = CocoDetectionpt(root=root, annFile=annotation_file)
    image_np, target_np = dataset_np[0]
    image_pt, target_pt = dataset_pt[0]
    assert image_np == image_pt
    assert target_np == target_pt
