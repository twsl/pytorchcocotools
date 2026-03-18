import torch
from torch import Tensor
from torchvision import tv_tensors as tv
from torchvision.ops.boxes import box_iou
from torchvision.transforms.v2 import functional as F  # noqa: N812


@torch.compile(dynamic=True, mode="reduce-overhead")
def _bb_iou_crowd(dt_xyxy: Tensor, gt_xyxy: Tensor, iscrowd_t: Tensor) -> Tensor:
    """Compute IoU with crowd support. Fully compilable kernel."""
    dt_area = (dt_xyxy[:, 2] - dt_xyxy[:, 0]) * (dt_xyxy[:, 3] - dt_xyxy[:, 1])
    gt_area = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])

    intersect_min = torch.max(dt_xyxy[:, None, :2], gt_xyxy[:, :2])
    intersect_max = torch.min(dt_xyxy[:, None, 2:], gt_xyxy[:, 2:])
    intersect_wh = torch.clamp(intersect_max - intersect_min, min=0)
    intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]

    union_area = dt_area[:, None] + gt_area - intersect_area
    union_area[:, iscrowd_t] = dt_area[:, None]
    return intersect_area / union_area


@torch.inference_mode()
def bbIou(dt: tv.BoundingBoxes, gt: tv.BoundingBoxes, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between bounding boxes.

    Args:
        dt: Detection bounding boxes (shape: [m, 4]).
        gt: Ground truth bounding boxes (shape: [n, 4]).
        iscrowd: List indicating if a ground truth bounding box is a crowd.

    Returns:
        IoU values for each detection and ground truth pair (shape: [m, n]).
    """
    # Convert to XYXY format (uses tv_tensor API, stays in Python wrapper)
    dt_xyxy = F.convert_bounding_box_format(dt, new_format=tv.BoundingBoxFormat.XYXY)
    gt_xyxy = F.convert_bounding_box_format(gt, new_format=tv.BoundingBoxFormat.XYXY)

    any_crowd = any(isc for isc in iscrowd)
    if not any_crowd:
        # box_iou accepts plain tensors
        return box_iou(dt_xyxy.as_subclass(Tensor), gt_xyxy.as_subclass(Tensor))

    iscrowd_t = torch.tensor(iscrowd, device=dt.device)
    return _bb_iou_crowd(
        dt_xyxy.as_subclass(Tensor),
        gt_xyxy.as_subclass(Tensor),
        iscrowd_t,
    )
