import json
from pathlib import Path

import pytest

from pytorchcocotools.internal.entities import IoUType
from pytorchcocotools.utils.generation import generate_mock_data


@pytest.fixture(scope="session", params=["bbox", "segm", "keypoints"])
def e2e_files(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[IoUType, Path, Path]:
    """Write a synthetic COCO gt/result pair to disk and return ``(iou_type, gt_path, dt_path)``."""
    iou_type: IoUType = request.param
    dataset, detections = generate_mock_data(iou_type)
    directory = tmp_path_factory.mktemp(f"e2e_{iou_type}")
    gt_path = directory / "gt.json"
    dt_path = directory / "dt.json"
    gt_path.write_text(json.dumps(dataset))
    dt_path.write_text(json.dumps(detections))
    return iou_type, gt_path, dt_path
