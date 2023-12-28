from pycocotools.cocoeval import COCOeval
import pytest
from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811
import torch


def test_computeIoU(coco_eval_np: COCOeval, coco_eval_pt: COCOevalpt):  # noqa: N802
    # test iou computation for keypoints
    ious = coco_eval.computeIoU(1, 1)
    expected_ious = torch.tensor([0.0, 0.0])
    assert torch.allclose(ious, expected_ious)

    # test iou computation for bounding boxes
    coco_eval.params.iouType = "bbox"
    ious = coco_eval.computeIoU(1, 1)
    expected_ious = torch.tensor([0.0, 0.0])
    assert torch.allclose(ious, expected_ious)
