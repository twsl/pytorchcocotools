from pycocotools.cocoeval import COCOeval
import pytest
import torch


@pytest.fixture
def coco_eval():
    # create a dummy COCO dataset
    dataset = {
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
    cocoGt = COCOeval.createGt(dataset)
    cocoDt = COCOeval.createDt(dataset)
    return COCOeval(cocoGt, cocoDt, "keypoints")


def test_computeIoU(coco_eval: COCOeval):  # noqa: N802
    # test iou computation for keypoints
    ious = coco_eval.computeIoU(1, 1)
    expected_ious = torch.tensor([0.0, 0.0])
    assert torch.allclose(ious, expected_ious)

    # test iou computation for bounding boxes
    coco_eval.params.iouType = "bbox"
    ious = coco_eval.computeIoU(1, 1)
    expected_ious = torch.tensor([0.0, 0.0])
    assert torch.allclose(ious, expected_ious)
