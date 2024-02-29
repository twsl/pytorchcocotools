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
                "num_keypoints": 5,
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10, 10, 20, 20],
                "area": 100,
                "iscrowd": 0,
                "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "num_keypoints": 5,
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
            },
        ],
        "images": [
            {
                "id": 1,
                "height": 50,
                "width": 100,
            }
        ],
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
                "num_keypoints": 5,
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
                "score": 0.5,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10, 10, 15, 15],
                "area": 100,
                "iscrowd": 0,
                "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "num_keypoints": 5,
                "segmentation": [[10, 10, 25, 10, 30, 30, 10, 30]],
                "score": 0.8,
            },
        ],
        "images": [
            {
                "id": 1,
                "height": 50,
                "width": 100,
            }
        ],
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
    eval = COCOevalnp(coco_gt_np, coco_dt_np, "bbox")
    eval._prepare()
    return eval


@fixture(scope="session")
def eval_bbox_pt(coco_gt_pt: COCOpt, coco_dt_pt: COCOpt) -> COCOevalpt:
    eval = COCOevalpt(coco_gt_pt, coco_dt_pt, "bbox")
    eval._prepare()
    return eval


@fixture(scope="session")
def eval_segm_np(coco_gt_np: COCOnp, coco_dt_np: COCOnp) -> COCOevalnp:
    eval = COCOevalnp(coco_gt_np, coco_dt_np, "segm")
    eval._prepare()
    return eval


@fixture(scope="session")
def eval_segm_pt(coco_gt_pt: COCOpt, coco_dt_pt: COCOpt) -> COCOevalpt:
    eval = COCOevalpt(coco_gt_pt, coco_dt_pt, "segm")
    eval._prepare()
    return eval


@fixture(scope="session")
def eval_keypoints_np(coco_gt_np: COCOnp, coco_dt_np: COCOnp) -> COCOevalnp:
    eval = COCOevalnp(coco_gt_np, coco_dt_np, "keypoints")
    eval._prepare()
    return eval


@fixture(scope="session")
def eval_keypoints_pt(coco_gt_pt: COCOpt, coco_dt_pt: COCOpt) -> COCOevalpt:
    eval = COCOevalpt(coco_gt_pt, coco_dt_pt, "keypoints")
    eval._prepare()
    return eval
