import torch
from torch import Tensor

from pytorchcocotools.internal.entities import BB, RLE, Mask, RLEs


def bbIou(dt: BB, gt: BB, m: int, n: int, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between bounding boxes.

    Args:
        dt: Detection bounding boxes (shape: [m, 4]).
        gt: Ground truth bounding boxes (shape: [n, 4]).
        m: Number of detection bounding boxes.
        n: Number of ground truth bounding boxes.
        iscrowd: List indicating if a ground truth bounding box is a crowd.

    Returns:
        IoU values for each detection and ground truth pair (shape: [m, n]).
    """
    # Convert the bounding boxes from [x1, y1, width, height] to [x1, y1, x2, y2]
    dt = torch.cat((dt[:, :2], dt[:, :2] + dt[:, 2:]), dim=1)
    gt = torch.cat((gt[:, :2], gt[:, :2] + gt[:, 2:]), dim=1)

    # Calculate area for detection and ground truth boxes
    dt_area = (dt[:, 2] - dt[:, 0]) * (dt[:, 3] - dt[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    # Compute intersection
    intersect_min = torch.max(dt[:, None, :2], gt[:, :2])  # [m, n, 2]
    intersect_max = torch.min(dt[:, None, 2:], gt[:, 2:])  # [m, n, 2]
    intersect_wh = torch.clamp(intersect_max - intersect_min, min=0)  # [m, n, 2]
    intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]  # [m, n]

    # Compute union
    union_area = dt_area[:, None] + gt_area - intersect_area
    union_area[torch.tensor(iscrowd)] = dt_area[:, None]  # Adjust for crowd

    # Compute IoU
    iou = intersect_area / union_area

    return iou
