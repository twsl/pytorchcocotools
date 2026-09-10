"""Adapted edge-case parity tests from hotcoco's scripts/test_parity.py."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from pycocotools.coco import COCO as COCOnp  # noqa: N811
from pycocotools.cocoeval import COCOeval as COCOevalnp  # noqa: N811
import pytest
import torch
from torchvision import tv_tensors as tv

from pytorchcocotools import mask
from pytorchcocotools.coco import COCO as COCOpt  # noqa: N811
from pytorchcocotools.cocoeval import COCOeval as COCOevalpt  # noqa: N811
from pytorchcocotools.internal.entities import IoUType


def _dataset(
    iou_type: IoUType, annotations: list[dict[str, Any]], images: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if images is None:
        images = [{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg"}]
    if iou_type == "keypoints":
        categories = [{"id": 1, "name": "person", "keypoints": [str(i) for i in range(17)], "skeleton": []}]
    else:
        categories = [{"id": 1, "name": "cat", "supercategory": "none"}]
    return {"images": images, "annotations": annotations, "categories": categories}


def _ann(ann_id: int, bbox: list[float], *, image_id: int = 1, iscrowd: int = 0, **extra: Any) -> dict[str, Any]:
    result = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": iscrowd,
    }
    result.update(extra)
    return result


def _det(bbox: list[float], *, image_id: int = 1, score: float = 0.9, **extra: Any) -> dict[str, Any]:
    return {"image_id": image_id, "category_id": 1, "bbox": bbox, "score": score, **extra}


def _stats(
    dataset: dict[str, Any], detections: list[dict[str, Any]], iou_type: IoUType
) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory() as directory:
        gt_path = Path(directory) / "gt.json"
        dt_path = Path(directory) / "dt.json"
        gt_path.write_text(json.dumps(dataset))
        dt_path.write_text(json.dumps(detections))

        coco_np = COCOnp(str(gt_path))
        eval_np = COCOevalnp(coco_np, coco_np.loadRes(str(dt_path)), iou_type)
        eval_np.evaluate()
        eval_np.accumulate()

        coco_pt = COCOpt(str(gt_path), enable_logging=False)
        eval_pt = COCOevalpt(coco_pt, coco_pt.loadRes(str(dt_path)), iou_type, enable_logging=False)
        eval_pt.evaluate()
        eval_pt.accumulate()

        return np.asarray(eval_np.stats), eval_pt.stats.detach().cpu().numpy()


def test_upstream_empty_gt_and_all_crowd_cases_match_reference() -> None:
    empty = _dataset("bbox", [])
    np_stats, pt_stats = _stats(empty, [_det([10, 10, 100, 100], score=0.5)], "bbox")
    np.testing.assert_allclose(np_stats, pt_stats)
    assert np.all(pt_stats[:6] == -1.0)

    crowd = _dataset("bbox", [_ann(1, [10, 10, 100, 100], iscrowd=1), _ann(2, [200, 200, 50, 50], iscrowd=1)])
    np_stats, pt_stats = _stats(crowd, [_det([10, 10, 100, 100]), _det([200, 200, 50, 50], score=0.5)], "bbox")
    np.testing.assert_allclose(np_stats, pt_stats)
    assert np.all(pt_stats == -1.0)


@pytest.mark.parametrize(
    ("annotations", "detections"),
    [
        (
            [_ann(i, [50, 50, 200, 150]) for i in range(1, 4)],
            [_det([50, 50, 200, 150], score=0.3 + i * 0.3) for i in range(3)],
        ),
        (
            [_ann(1, [10, 10, 32, 32]), _ann(2, [100, 100, 96, 96])],
            [_det([10, 10, 32, 32]), _det([100, 100, 96, 96], score=0.7)],
        ),
        (
            [_ann(1, [10, 10, 0, 50]), _ann(2, [50, 50, 50, 0]), _ann(3, [100, 100, 80, 80])],
            [_det([10, 10, 0, 50]), _det([50, 50, 50, 0], score=0.8), _det([100, 100, 80, 80], score=0.7)],
        ),
    ],
)
def test_upstream_bbox_edge_cases_match_reference(
    annotations: list[dict[str, Any]], detections: list[dict[str, Any]]
) -> None:
    np_stats, pt_stats = _stats(_dataset("bbox", annotations), detections, "bbox")
    np.testing.assert_allclose(np_stats, pt_stats)


def test_upstream_segmentation_and_keypoint_cases_match_reference() -> None:
    segmentation = [[10, 10, 110, 10, 110, 110, 10, 110]]
    dataset = _dataset("segm", [_ann(1, [10, 10, 100, 100], segmentation=segmentation)])
    np_stats, pt_stats = _stats(dataset, [_det([10, 10, 100, 100], segmentation=segmentation)], "segm")
    np.testing.assert_allclose(np_stats, pt_stats)

    keypoints = [float(i) for i in range(51)]
    dataset = _dataset("keypoints", [_ann(1, [50, 50, 200, 200], keypoints=keypoints, num_keypoints=17)])
    detection = {"image_id": 1, "category_id": 1, "score": 0.9, "keypoints": keypoints}
    np_stats, pt_stats = _stats(dataset, [detection], "keypoints")
    np.testing.assert_allclose(np_stats, pt_stats)


def test_upstream_rle_counts_forms_round_trip() -> None:
    array = tv.Mask(torch.zeros((10, 10), dtype=torch.uint8))
    array[2:5, 2:5] = 1
    encoded = mask.encode(array)
    assert mask.decode(encoded).sum().item() == 9
