from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import IntTensor, Tensor
from torchmetrics.detection import MeanAveragePrecision as TorchmetricsMeanAveragePrecision

from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision


@pytest.fixture()
def inputs():
    return {
        "preds": [
            [
                {
                    "boxes": Tensor([[258.15, 41.29, 606.41, 285.07]]),
                    "scores": Tensor([0.236]),
                    "labels": IntTensor([4]),
                },
                {
                    "boxes": Tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
                    "scores": Tensor([0.318, 0.726]),
                    "labels": IntTensor([3, 2]),
                },
            ],
            [
                {
                    "boxes": Tensor(
                        [
                            [87.87, 276.25, 384.29, 379.43],
                            [0.00, 3.66, 142.15, 316.06],
                            [296.55, 93.96, 314.97, 152.79],
                            [328.94, 97.05, 342.49, 122.98],
                            [356.62, 95.47, 372.33, 147.55],
                            [464.08, 105.09, 495.74, 146.99],
                            [276.11, 103.84, 291.44, 150.72],
                        ]
                    ),
                    "scores": Tensor([0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953]),
                    "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
                },
                {
                    "boxes": Tensor(
                        [
                            [72.92, 45.96, 91.23, 80.57],
                            [45.17, 45.34, 66.28, 79.83],
                            [82.28, 47.04, 99.66, 78.50],
                            [59.96, 46.17, 80.35, 80.48],
                            [75.29, 23.01, 91.85, 50.85],
                            [71.14, 1.10, 96.96, 28.33],
                            [61.34, 55.23, 77.14, 79.57],
                            [41.17, 45.78, 60.99, 78.48],
                            [56.18, 44.80, 64.42, 56.25],
                        ]
                    ),
                    "scores": Tensor([0.532, 0.204, 0.782, 0.202, 0.883, 0.271, 0.561, 0.204, 0.349]),
                    "labels": IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49]),
                },
            ],
        ],
        "target": [
            [
                {
                    "boxes": Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                    "labels": IntTensor([4]),
                },
                {
                    "boxes": Tensor([[13.00, 22.75, 548.98, 632.42], [1.66, 3.32, 270.26, 275.23]]),
                    "labels": IntTensor([2, 2]),
                },
            ],
            [
                {
                    "boxes": Tensor(
                        [
                            [61.87, 276.25, 358.29, 379.43],
                            [2.75, 3.66, 162.15, 316.06],
                            [295.55, 93.96, 313.97, 152.79],
                            [326.94, 97.05, 340.49, 122.98],
                            [356.62, 95.47, 372.33, 147.55],
                            [462.08, 105.09, 493.74, 146.99],
                            [277.11, 103.84, 292.44, 150.72],
                        ]
                    ),
                    "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
                },
                {
                    "boxes": Tensor(
                        [
                            [72.92, 45.96, 91.23, 80.57],
                            [50.17, 45.34, 71.28, 79.83],
                            [81.28, 47.04, 98.66, 78.50],
                            [63.96, 46.17, 84.35, 80.48],
                            [75.29, 23.01, 91.85, 50.85],
                            [56.39, 21.65, 75.66, 45.54],
                            [73.14, 1.10, 98.96, 28.33],
                            [62.34, 55.23, 78.14, 79.57],
                            [44.17, 45.78, 63.99, 78.48],
                            [58.18, 44.80, 66.42, 56.25],
                        ]
                    ),
                    "labels": IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49, 49]),
                },
            ],
        ],
    }


@pytest.fixture()
def inputs2():
    return {
        "preds": [
            [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": Tensor([0.536]), "labels": IntTensor([0])}],
            [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": Tensor([0.536]), "labels": IntTensor([0])}],
        ],
        "target": [
            [{"boxes": Tensor([[214.0, 41.0, 562.0, 285.0]]), "labels": IntTensor([0])}],
            [{"boxes": Tensor([]), "labels": IntTensor([])}],
        ],
    }


@pytest.fixture()
def inputs3():
    return {
        "preds": [
            [{"boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": Tensor([0.536]), "labels": IntTensor([0])}],
            [{"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
        ],
        "target": [
            [{"boxes": Tensor([[214.0, 41.0, 562.0, 285.0]]), "labels": IntTensor([0])}],
            [{"boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]), "labels": IntTensor([1])}],
        ],
    }


@pytest.fixture()
def tm_compute() -> Callable[..., dict[str, Tensor]]:
    def _compute(preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]], **kwargs: Any) -> dict[str, Tensor]:
        m = TorchmetricsMeanAveragePrecision(**kwargs)
        m.update(preds, target)
        return m.compute()

    return _compute


@pytest.fixture()
def pt_compute() -> Callable[..., dict[str, Tensor]]:
    def _compute(preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]], **kwargs: Any) -> dict[str, Tensor]:
        m = MeanAveragePrecision(**kwargs)
        m.update(preds, target)
        return m.compute()

    return _compute


@pytest.fixture()
def assert_map_close() -> Callable[[dict[str, Tensor], dict[str, Tensor], float], None]:
    def _assert(result: dict[str, Tensor], reference: dict[str, Tensor], atol: float = 1e-4) -> None:
        scalar_keys = [k for k in reference if k != "classes" and reference[k].ndim == 0]
        for key in scalar_keys:
            assert key in result, f"Missing key '{key}' in result"
            torch.testing.assert_close(
                result[key].float(),
                reference[key].float(),
                atol=atol,
                rtol=0.0,
                msg=f"Mismatch for key '{key}'",
            )

    return _assert


@pytest.fixture()
def make_random_boxes() -> Callable[[int, float, float, int], Tensor]:
    def _make(n: int, img_w: float = 640.0, img_h: float = 480.0, seed: int = 0) -> Tensor:
        """Return *n* random non-degenerate xyxy boxes inside (img_w x img_h)."""
        gen = torch.Generator().manual_seed(seed)
        x1 = torch.rand(n, generator=gen) * (img_w * 0.8)
        y1 = torch.rand(n, generator=gen) * (img_h * 0.8)
        w = torch.rand(n, generator=gen) * (img_w * 0.2) + 10.0
        h = torch.rand(n, generator=gen) * (img_h * 0.2) + 10.0
        x2 = (x1 + w).clamp(max=img_w)
        y2 = (y1 + h).clamp(max=img_h)
        return torch.stack([x1, y1, x2, y2], dim=1)

    return _make


@pytest.fixture()
def make_stress_batch(
    make_random_boxes: Callable[[int, float, float, int], Tensor],
) -> Callable[[int, int, int, int], tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]]:
    def _make(
        n_images: int,
        n_boxes_per_image: int,
        n_classes: int = 10,
        seed: int = 42,
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        """Build preds + target lists with *n_boxes_per_image* boxes in each image."""
        preds, target = [], []
        for i in range(n_images):
            boxes = make_random_boxes(n_boxes_per_image, seed=seed + i)
            labels = torch.randint(
                0, n_classes, (n_boxes_per_image,), generator=torch.Generator().manual_seed(seed + i + 1000)
            )
            preds.append(
                {
                    "boxes": boxes,
                    "scores": torch.rand(n_boxes_per_image, generator=torch.Generator().manual_seed(seed + i + 2000)),
                    "labels": labels.to(torch.int32),
                }
            )
            gt_boxes = make_random_boxes(n_boxes_per_image, seed=seed + i + 3000)
            gt_labels = torch.randint(
                0, n_classes, (n_boxes_per_image,), generator=torch.Generator().manual_seed(seed + i + 4000)
            )
            target.append(
                {
                    "boxes": gt_boxes,
                    "labels": gt_labels.to(torch.int32),
                }
            )
        return preds, target

    return _make
