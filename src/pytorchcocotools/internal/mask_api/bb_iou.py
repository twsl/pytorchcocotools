import torch
from torch import Tensor
from torchvision import tv_tensors as tv
from torchvision.transforms.v2 import functional as F  # noqa: N812


@torch.compile(dynamic=True, mode="reduce-overhead")
def _bb_iou_core(dt_xyxy: Tensor, gt_xyxy: Tensor, iscrowd_t: Tensor | None) -> Tensor:
    """Compiled kernel for all-pairs bounding-box IoU.

    Note:
        implementation equal to `torchvision.ops.boxes.box_iou` but with optional crowd handling and fused arithmetic for speed.

    Args:
         dt_xyxy: Detection boxes in XYXY format (shape: [m, 4]).
         gt_xyxy: Ground truth boxes in XYXY format (shape: [n, 4]).
         iscrowd_t: Optional boolean tensor indicating which GT boxes are crowds (shape: [n]).

    Returns:
         IoU values for each detection and ground truth pair (shape: [m, n]).
    """
    dt_area = (dt_xyxy[:, 2] - dt_xyxy[:, 0]) * (dt_xyxy[:, 3] - dt_xyxy[:, 1])
    gt_area = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])

    intersect_min = torch.max(dt_xyxy[:, None, :2], gt_xyxy[None, :, :2])
    intersect_max = torch.min(dt_xyxy[:, None, 2:], gt_xyxy[None, :, 2:])
    intersect_wh = (intersect_max - intersect_min).clamp(min=0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    union_area = dt_area[:, None] + gt_area[None, :] - intersect_area

    if iscrowd_t is not None:
        union_area = torch.where(iscrowd_t, dt_area[:, None], union_area)

    return intersect_area / union_area


@torch.inference_mode()
def bbIou(dt: tv.BoundingBoxes, gt: tv.BoundingBoxes, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between bounding boxes.

    Converts ``tv.BoundingBoxes`` to plain tensors in eager mode, then
    delegates to a ``torch.compile``-d kernel that fuses the element-wise
    arithmetic into a single pass for 2-4x speedup.

    Args:
        dt: Detection bounding boxes (shape: [m, 4]).
        gt: Ground truth bounding boxes (shape: [n, 4]).
        iscrowd: List indicating if a ground truth bounding box is a crowd.

    Returns:
        IoU values for each detection and ground truth pair (shape: [m, n]).
    """
    dt_xyxy = F.convert_bounding_box_format(dt, new_format=tv.BoundingBoxFormat.XYXY).as_subclass(Tensor)
    gt_xyxy = F.convert_bounding_box_format(gt, new_format=tv.BoundingBoxFormat.XYXY).as_subclass(Tensor)

    iscrowd_t = torch.tensor(iscrowd, dtype=torch.bool, device=dt.device) if any(iscrowd) else None

    return _bb_iou_core(dt_xyxy, gt_xyxy, iscrowd_t)
