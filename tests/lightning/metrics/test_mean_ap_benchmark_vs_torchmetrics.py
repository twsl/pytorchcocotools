import pytest
from pytest_benchmark.fixture import BenchmarkFixture
import torch


def _import_metrics():
    pytest.importorskip("torchmetrics")
    pytest.importorskip("lightning_utilities")
    from torchmetrics.detection.mean_ap import MeanAveragePrecision as TMMeanAP

    from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision as PTMeanAP

    return TMMeanAP, PTMeanAP


def _make_workload(
    *,
    num_images: int = 50,
    num_classes: int = 5,
    gts_per_image: int = 10,
    preds_per_image: int = 20,
    seed: int = 123,
):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    preds = []
    target = []

    for _img_idx in range(num_images):
        labels_t = torch.randint(0, num_classes, (gts_per_image,), generator=g, dtype=torch.int64)
        xy1 = torch.rand((gts_per_image, 2), generator=g) * 256
        wh = torch.rand((gts_per_image, 2), generator=g) * 64 + 1
        xy2 = xy1 + wh
        boxes_t = torch.cat([xy1, xy2], dim=1).to(dtype=torch.float32)

        target.append(
            {
                "boxes": boxes_t,
                "labels": labels_t,
                "iscrowd": torch.zeros((gts_per_image,), dtype=torch.int32),
            }
        )

        labels_p = torch.randint(0, num_classes, (preds_per_image,), generator=g, dtype=torch.int64)
        scores_p = torch.rand((preds_per_image,), generator=g).to(dtype=torch.float32)
        xy1p = torch.rand((preds_per_image, 2), generator=g) * 256
        whp = torch.rand((preds_per_image, 2), generator=g) * 64 + 1
        xy2p = xy1p + whp
        boxes_p = torch.cat([xy1p, xy2p], dim=1).to(dtype=torch.float32)

        preds.append(
            {
                "boxes": boxes_p,
                "labels": labels_p,
                "scores": scores_p,
            }
        )

    return preds, target


def _boxes_to_masks(boxes: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    masks = torch.zeros((boxes.shape[0], height, width), dtype=torch.bool)
    for i, box in enumerate(boxes):
        x1 = int(torch.floor(box[0]).item())
        y1 = int(torch.floor(box[1]).item())
        x2 = int(torch.ceil(box[2]).item())
        y2 = int(torch.ceil(box[3]).item())

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        masks[i, y1:y2, x1:x2] = True
    return masks


def _make_workload_with_masks(
    *,
    num_images: int = 60,
    num_classes: int = 7,
    gts_per_image: int = 14,
    preds_per_image: int = 28,
    seed: int = 123,
    height: int = 384,
    width: int = 384,
):
    preds, target = _make_workload(
        num_images=num_images,
        num_classes=num_classes,
        gts_per_image=gts_per_image,
        preds_per_image=preds_per_image,
        seed=seed,
    )

    for p, t in zip(preds, target, strict=True):
        t["masks"] = _boxes_to_masks(t["boxes"], height=height, width=width)
        p["masks"] = _boxes_to_masks(p["boxes"], height=height, width=width)

    return preds, target


@pytest.mark.benchmark(group="mean_ap", warmup=True)
def test_mean_ap_compute_np(benchmark: BenchmarkFixture) -> None:
    tm_mean_ap, _pt_mean_ap = _import_metrics()
    preds, target = _make_workload(seed=321)

    metric = tm_mean_ap(iou_type="bbox", box_format="xyxy", backend="pycocotools", class_metrics=False)
    metric.update(preds, target)

    def _run():
        metric._computed = None  # torchmetrics cache
        return metric.compute()

    benchmark(_run)


@pytest.mark.benchmark(group="mean_ap", warmup=True)
def test_mean_ap_compute_pt(benchmark: BenchmarkFixture) -> None:
    _tm_mean_ap, pt_mean_ap = _import_metrics()
    preds, target = _make_workload(seed=321)

    metric = pt_mean_ap(iou_type="bbox", box_format="xyxy", class_metrics=False)
    metric.update(preds, target)

    def _run():
        metric._computed = None  # torchmetrics cache
        return metric.compute()

    benchmark(_run)


@pytest.mark.benchmark(group="mean_ap_class", warmup=True)
def test_mean_ap_class_metrics_np(benchmark: BenchmarkFixture) -> None:
    tm_mean_ap, _pt_mean_ap = _import_metrics()
    preds, target = _make_workload(seed=777)

    metric = tm_mean_ap(iou_type="bbox", box_format="xyxy", backend="pycocotools", class_metrics=True)
    metric.update(preds, target)

    def _run():
        metric._computed = None  # torchmetrics cache
        return metric.compute()

    benchmark(_run)


@pytest.mark.benchmark(group="mean_ap_class", warmup=True)
def test_mean_ap_class_metrics_pt(benchmark: BenchmarkFixture) -> None:
    _tm_mean_ap, pt_mean_ap = _import_metrics()
    preds, target = _make_workload(seed=777)

    metric = pt_mean_ap(iou_type="bbox", box_format="xyxy", class_metrics=True)
    metric.update(preds, target)

    def _run():
        metric._computed = None  # torchmetrics cache
        return metric.compute()

    benchmark(_run)


@pytest.mark.benchmark(group="mean_ap_complex", warmup=True)
def test_mean_ap_compute_complex_segm_np(benchmark: BenchmarkFixture) -> None:
    tm_mean_ap, _pt_mean_ap = _import_metrics()
    preds, target = _make_workload_with_masks(seed=2026)

    metric = tm_mean_ap(
        iou_type="segm",
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True,
        average="micro",
        backend="pycocotools",
    )
    metric.update(preds, target)

    def _run():
        metric._computed = None  # torchmetrics cache
        return metric.compute()

    benchmark(_run)


@pytest.mark.benchmark(group="mean_ap_complex", warmup=True)
def test_mean_ap_compute_complex_segm_pt(benchmark: BenchmarkFixture) -> None:
    _tm_mean_ap, pt_mean_ap = _import_metrics()
    preds, target = _make_workload_with_masks(seed=2026)

    metric = pt_mean_ap(
        iou_type="segm",
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True,
        average="micro",
    )
    metric.update(preds, target)

    def _run():
        metric._computed = None  # torchmetrics cache
        return metric.compute()

    benchmark(_run)


@pytest.mark.benchmark(group="mean_ap_complex_mixed", warmup=True)
def test_mean_ap_compute_complex_bbox_segm_np(benchmark: BenchmarkFixture) -> None:
    tm_mean_ap, _pt_mean_ap = _import_metrics()
    preds, target = _make_workload_with_masks(seed=1337)

    metric = tm_mean_ap(
        box_format="xyxy",
        iou_type=("bbox", "segm"),
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True,
        average="micro",
        backend="pycocotools",
    )
    metric.update(preds, target)

    def _run():
        metric._computed = None  # torchmetrics cache
        return metric.compute()

    benchmark(_run)


@pytest.mark.benchmark(group="mean_ap_complex_mixed", warmup=True)
def test_mean_ap_compute_complex_bbox_segm_pt(benchmark: BenchmarkFixture) -> None:
    _tm_mean_ap, pt_mean_ap = _import_metrics()
    preds, target = _make_workload_with_masks(seed=1337)

    metric = pt_mean_ap(
        box_format="xyxy",
        iou_type=("bbox", "segm"),
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True,
        average="micro",
    )
    metric.update(preds, target)

    def _run():
        metric._computed = None  # torchmetrics cache
        return metric.compute()

    benchmark(_run)
