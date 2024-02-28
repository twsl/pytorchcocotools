from pycocotools.coco import COCO as COCOnp  # noqa: N811
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
from pytest_cases import parametrize_with_cases
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811
from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811  # noqa: N811
import torch

from .base_cases import BaseCases


class COCOEvalCases(BaseCases):
    def _eval_bbox(
        self, type: COCOevalnp | COCOevalpt, coco_gt: COCOnp | COCOpt, coco_dt: COCOnp | COCOpt
    ) -> COCOevalnp | COCOevalpt:
        return type(coco_gt, coco_dt, "bbox")

    def case_eval_bbox_np(self, coco_gt_np: COCOnp, coco_dt_np: COCOnp) -> tuple:
        eval = self._eval_bbox(COCOevalnp, coco_gt_np, coco_dt_np)
        return (eval, 1, 1, [0.0])

    def case_eval_bbox_pt(self, coco_gt_pt: COCOpt, coco_dt_pt: COCOpt) -> tuple:
        eval = self._eval_bbox(COCOevalpt, coco_gt_pt, coco_dt_pt)
        return (eval, 1, 1, [0.0])

    def case_eval_bbox_both(
        self, coco_gt_np: COCOnp, coco_dt_np: COCOnp, coco_gt_pt: COCOpt, coco_dt_pt: COCOpt
    ) -> tuple:
        eval_np = self._eval_bbox(COCOevalnp, coco_gt_np, coco_dt_np)
        eval_pt = self._eval_bbox(COCOevalpt, coco_gt_pt, coco_dt_pt)
        return (eval_np, eval_pt, 1, 1, [0.0])

    def _eval_segm(
        self, type: COCOevalnp | COCOevalpt, coco_gt: COCOnp | COCOpt, coco_dt: COCOnp | COCOpt
    ) -> COCOevalnp | COCOevalpt:
        return type(coco_gt, coco_dt, "segm")

    def case_eval_segm_np(self, coco_gt_np: COCOnp, coco_dt_np: COCOnp) -> tuple:
        eval = self._eval_segm(COCOevalnp, coco_gt_np, coco_dt_np)
        return (eval, 1, 1, [0.0])

    def case_eval_segm_pt(self, coco_gt_pt: COCOpt, coco_dt_pt: COCOpt) -> tuple:
        eval = self._eval_segm(COCOevalpt, coco_gt_pt, coco_dt_pt)
        return (eval, 1, 1, [0.0])

    def case_eval_segm_both(
        self, coco_gt_np: COCOnp, coco_dt_np: COCOnp, coco_gt_pt: COCOpt, coco_dt_pt: COCOpt
    ) -> tuple:
        eval_np = self._eval_segm(COCOevalnp, coco_gt_np, coco_dt_np)
        eval_pt = self._eval_segm(COCOevalpt, coco_gt_pt, coco_dt_pt)
        return (eval_np, eval_pt, 1, 1, [0.0])

    def _eval_keypoints(
        self, type: COCOevalnp | COCOevalpt, coco_gt: COCOnp | COCOpt, coco_dt: COCOnp | COCOpt
    ) -> COCOevalnp | COCOevalpt:
        return type(coco_gt, coco_dt, "keypoints")

    def case_eval_keypoints_np(self, coco_gt_np: COCOnp, coco_dt_np: COCOnp) -> tuple:
        eval = self._eval_keypoints(COCOevalnp, coco_gt_np, coco_dt_np)
        return (eval, 1, 1, [0.0])

    def case_eval_keypoints_pt(self, coco_gt_pt: COCOpt, coco_dt_pt: COCOpt) -> tuple:
        eval = self._eval_keypoints(COCOevalpt, coco_gt_pt, coco_dt_pt)
        return (eval, 1, 1, [0.0])

    def case_eval_keypoints_both(
        self, coco_gt_np: COCOnp, coco_dt_np: COCOnp, coco_gt_pt: COCOpt, coco_dt_pt: COCOpt
    ) -> tuple:
        eval_np = self._eval_keypoints(COCOevalnp, coco_gt_np, coco_dt_np)
        eval_pt = self._eval_keypoints(COCOevalpt, coco_gt_pt, coco_dt_pt)
        return (eval_np, eval_pt, 1, 1, [0.0])


@pytest.mark.benchmark(group="computeIoU", warmup=True)
@parametrize_with_cases("coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCases, glob="*pt")
def test_computeIoU_pt(benchmark, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result):  # noqa: N802
    ious = coco_eval_pt.computeIoU(img_id, cat_id)
    # ious = benchmark(coco_eval_pt.computeIoU, img_id, cat_id)
    assert torch.allclose(ious, result)


@pytest.mark.benchmark(group="computeIoU", warmup=True)
@parametrize_with_cases("coco_eval_np, img_id, cat_id, result", cases=COCOEvalCases, glob="*np")
def test_computeIoU_np(benchmark, coco_eval_np: COCOevalnp, img_id: int, cat_id: int, result):  # noqa: N802
    ious = coco_eval_np.computeIoU(img_id, cat_id)
    # ious = benchmark(coco_eval_np.computeIoU, img_id, cat_id)
    assert torch.allclose(ious, result)


@parametrize_with_cases("coco_eval_np, coco_eval_pt, img_id, cat_id, result", cases=COCOEvalCases, glob="*both")
def test_computeIoU(coco_eval_np: COCOevalnp, coco_eval_pt: COCOevalpt, img_id: int, cat_id: int, result):  # noqa: N802
    ious_np = coco_eval_np.computeIoU(img_id, cat_id)
    ious_pt = coco_eval_pt.computeIoU(img_id, cat_id)
    assert torch.allclose(ious_np, ious_pt)
    assert torch.allclose(ious_pt, result)
