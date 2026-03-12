from pytest_cases import fixture
import torch
from torch import Tensor


@fixture
def bbox_preds() -> list[dict[str, Tensor]]:
    """Single-image bounding-box predictions."""
    return [
        {
            "boxes": torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
            "scores": torch.tensor([0.536]),
            "labels": torch.tensor([0]),
        }
    ]


@fixture
def bbox_target() -> list[dict[str, Tensor]]:
    """Single-image bounding-box ground-truths."""
    return [
        {
            "boxes": torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
            "labels": torch.tensor([0]),
        }
    ]


@fixture
def multi_image_preds() -> list[dict[str, Tensor]]:
    """Multi-image bounding-box predictions (two images, two classes)."""
    return [
        {
            "boxes": torch.tensor([[100.0, 50.0, 300.0, 200.0], [400.0, 100.0, 600.0, 300.0]]),
            "scores": torch.tensor([0.9, 0.75]),
            "labels": torch.tensor([0, 1]),
        },
        {
            "boxes": torch.tensor([[10.0, 10.0, 110.0, 110.0]]),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([0]),
        },
    ]


@fixture
def multi_image_target() -> list[dict[str, Tensor]]:
    """Multi-image bounding-box ground-truths."""
    return [
        {
            "boxes": torch.tensor([[102.0, 52.0, 298.0, 198.0], [398.0, 98.0, 602.0, 302.0]]),
            "labels": torch.tensor([0, 1]),
        },
        {
            "boxes": torch.tensor([[12.0, 12.0, 108.0, 108.0]]),
            "labels": torch.tensor([0]),
        },
    ]


@fixture
def segm_preds() -> list[dict[str, Tensor]]:
    """Single-image segmentation-mask predictions."""
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[1:3, 2:4] = True
    return [
        {
            "masks": mask.unsqueeze(0),
            "scores": torch.tensor([0.536]),
            "labels": torch.tensor([0]),
        }
    ]


@fixture
def segm_target() -> list[dict[str, Tensor]]:
    """Single-image segmentation-mask ground-truths."""
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[1:4, 2:3] = True
    mask[2, 2:4] = True
    return [
        {
            "masks": mask.unsqueeze(0),
            "labels": torch.tensor([0]),
        }
    ]
