from pycocotools.coco import COCO as COCO
from pytest_cases import fixture
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811


@fixture(scope="session")
def dataset_gt() -> dict:
    return {
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
                "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10, 10, 20, 20],
                "area": 100,
                "iscrowd": 0,
                "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
        ],
        "images": [{"id": 1}],
        "categories": [{"id": 1}, {"id": 2}],
    }


@fixture(scope="session")
def dataset_dt() -> dict:
    return {
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
                "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10, 10, 20, 20],
                "area": 100,
                "iscrowd": 0,
                "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
        ],
        "images": [{"id": 1}],
        "categories": [{"id": 1}, {"id": 2}],
    }


@fixture(scope="session")
def coco_gt_np(dataset_gt: dict) -> COCO:
    coco = COCO()
    coco.dataset = dataset_gt
    coco.createIndex()
    return coco


@fixture(scope="session")
def coco_dt_np(dataset_dt: dict) -> COCO:
    coco = COCO()
    coco.dataset = dataset_dt
    coco.createIndex()
    return coco


@fixture(scope="session")
def coco_gt_pt(dataset_gt: dict) -> COCOpt:
    coco = COCOpt()
    coco.dataset = dataset_gt
    coco.createIndex()
    return coco


@fixture(scope="session")
def coco_dt_pt(dataset_dt: dict) -> COCOpt:
    coco = COCOpt()
    coco.dataset = dataset_dt
    coco.createIndex()
    return coco
