from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_cases import parametrize_with_cases
import torch
from torchvision.datasets import CocoDetection as CocoDetectionnp
from torchvision.transforms.functional import pil_to_tensor

from pytorchcocotools.torch.dataset import CocoDetection as CocoDetectionpt


class DatasetCases:
    def case_example(self) -> tuple[str, str, int, int]:
        return ("./data/coco", "./data/example.json", 0, 1)


@pytest.mark.benchmark(group="dataset", warmup=True)
@parametrize_with_cases("root, annotation_file, index, result", cases=DatasetCases)
def test_dataset_np(benchmark: BenchmarkFixture, root: str, annotation_file: str, index: int, result: Any) -> None:
    benchmark.weave(CocoDetectionpt.__getitem__, lazy=True)
    dataset = CocoDetectionnp(root=root, annFile=annotation_file)
    image_np, target_np = dataset[index]
    # compare the results
    assert target_np[0]["category_id"] == result


@pytest.mark.benchmark(group="dataset", warmup=True)
@parametrize_with_cases("root, annotation_file, index, result", cases=DatasetCases)
def test_dataset_pt(benchmark: BenchmarkFixture, root: str, annotation_file: str, index: int, result: Any) -> None:
    benchmark.weave(CocoDetectionpt.__getitem__, lazy=True)
    dataset = CocoDetectionpt(root=root, annotation_path=annotation_file)
    image_pt, target_pt = dataset[index]
    # compare the results
    assert target_pt["labels"].item() == result


@parametrize_with_cases("root, annotation_file, index, result", cases=DatasetCases)
def test_dataset(root: str, annotation_file: str, index: int, result: Any) -> None:
    dataset_np = CocoDetectionnp(root=root, annFile=annotation_file)
    dataset_pt = CocoDetectionpt(root=root, annotation_path=annotation_file)
    image_np, target_np = dataset_np[index]
    image_pt, target_pt = dataset_pt[index]
    assert torch.allclose(pil_to_tensor(image_np), image_pt)
    # assert target_np[0]["segmentation"] == target_pt["masks"]
    # assert target_np[0]["bbox"] == target_pt["boxes"]
    # assert target_np[0]["image_id"] == target_pt["image_id"]
    # assert target_np[0]["area"] == target_pt["area"]
    # assert target_np[0]["iscrowd"] == target_pt["iscrowd"]
    assert target_np[0]["category_id"] == target_pt["labels"].item()
