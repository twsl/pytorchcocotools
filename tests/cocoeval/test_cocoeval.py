# import numpy as np
# from pycocotools.coco import COCO as COCO
# from pycocotools.cocoeval import COCOeval
# import pycocotools.mask as mask
# import pytest
# from pytest_cases import case, parametrize_with_cases
# from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811
# from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811  # noqa: N811
# import pytorchcocotools.mask as tmask
# import torch


# class COCOEvalCases:
#     def case_eval_bbox(self, coco_gt_np: COCO) -> tuple:
#         return COCOeval(coco_gt_np, coco_dt_np, "keypoints")


# @parametrize_with_cases("min, max, h, w, result", cases=COCOEvalCases)
# def test_computeIoU(coco_eval_np: COCOeval, coco_eval_pt: COCOevalpt):  # noqa: N802
#     # test iou computation for keypoints
#     ious = coco_eval_np.computeIoU(1, 1)
#     expected_ious = torch.tensor([0.0, 0.0])
#     assert torch.allclose(ious, expected_ious)

#     # test iou computation for bounding boxes
#     coco_eval_np.params.iouType = "bbox"
#     ious = coco_eval_np.computeIoU(1, 1)
#     expected_ious = torch.tensor([0.0, 0.0])
#     assert torch.allclose(ious, expected_ious)
