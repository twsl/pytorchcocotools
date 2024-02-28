from pycocotools.coco import COCO as COCOnp  # noqa: N811
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
from pytest_cases import fixture
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811
from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811


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
def coco_gt_np(dataset_gt: dict) -> COCOnp:
    coco = COCOnp()
    coco.dataset = dataset_gt
    coco.createIndex()
    return coco


@fixture(scope="session")
def coco_dt_np(dataset_dt: dict) -> COCOnp:
    coco = COCOnp()
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


@fixture(scope="session")
def eval_bbox_np(coco_gt_np: COCOnp, coco_dt_np: COCOnp) -> COCOevalnp:
    return COCOevalnp(coco_gt_np, coco_dt_np, "bbox")


@fixture(scope="session")
def eval_bbox_pt(coco_gt_pt: COCOpt, coco_dt_pt: COCOpt) -> COCOevalpt:
    return COCOevalpt(coco_gt_pt, coco_dt_pt, "bbox")


@fixture(scope="session")
def eval_segm_np(coco_gt_np: COCOnp, coco_dt_np: COCOnp) -> COCOevalnp:
    return COCOevalnp(coco_gt_np, coco_dt_np, "segm")


@fixture(scope="session")
def eval_segm_pt(coco_gt_pt: COCOpt, coco_dt_pt: COCOpt) -> COCOevalpt:
    return COCOevalpt(coco_gt_pt, coco_dt_pt, "segm")


@fixture(scope="session")
def eval_keypoints_np(coco_gt_np: COCOnp, coco_dt_np: COCOnp) -> COCOevalnp:
    return COCOevalnp(coco_gt_np, coco_dt_np, "keypoints")


@fixture(scope="session")
def eval_keypoints_pt(coco_gt_pt: COCOpt, coco_dt_pt: COCOpt) -> COCOevalpt:
    return COCOevalpt(coco_gt_pt, coco_dt_pt, "keypoints")
