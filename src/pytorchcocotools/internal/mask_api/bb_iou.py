import torch
from torch import Tensor
from torchvision import tv_tensors as tv
from torchvision.ops.boxes import box_convert, box_iou
from torchvision.transforms.v2 import functional as F  # noqa: N812


def bbIou(dt: tv.BoundingBoxes, gt: tv.BoundingBoxes, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between bounding boxes.

    Args:
        dt: Detection bounding boxes (shape: [m, 4]).
        gt: Ground truth bounding boxes (shape: [n, 4]).
        iscrowd: List indicating if a ground truth bounding box is a crowd.

    Returns:
        IoU values for each detection and ground truth pair (shape: [m, n]).
    """
    any_crowd = any(isc for isc in iscrowd)
    dt = tv.wrap(F.convert_bounding_box_format(dt, new_format=tv.BoundingBoxFormat.XYXY), like=dt)
    gt = tv.wrap(F.convert_bounding_box_format(gt, new_format=tv.BoundingBoxFormat.XYXY), like=gt)

    if not any_crowd:
        iou = box_iou(dt, gt)

    # else:

    # # Convert the bounding boxes from [x1, y1, width, height] to [x1, y1, x2, y2]
    # dt = torch.cat((dt[:, :2], dt[:, :2] + dt[:, 2:]), dim=1)
    # gt = torch.cat((gt[:, :2], gt[:, :2] + gt[:, 2:]), dim=1)

    # # Calculate area for detection and ground truth boxes
    # dt_area = (dt[:, 2] - dt[:, 0]) * (dt[:, 3] - dt[:, 1])
    # gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    # # Compute intersection
    # intersect_min = torch.max(dt[:, None, :2], gt[:, :2])  # [m, n, 2]
    # intersect_max = torch.min(dt[:, None, 2:], gt[:, 2:])  # [m, n, 2]
    # intersect_wh = torch.clamp(intersect_max - intersect_min, min=0)  # [m, n, 2]
    # intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]  # [m, n]

    # # Compute union
    # union_area = dt_area[:, None] + gt_area - intersect_area
    # union_area[torch.tensor(iscrowd)] = dt_area[:, None]  # Adjust for crowd

    # # Compute IoU
    # iou = intersect_area / union_area

    return iou
