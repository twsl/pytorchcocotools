from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch


def import_metrics():
    pytest.importorskip("torchmetrics")
    pytest.importorskip("lightning_utilities")
    from torchmetrics.detection.mean_ap import MeanAveragePrecision as TMMeanAP

    from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision as PTMeanAP

    return TMMeanAP, PTMeanAP


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@dataclass(frozen=True)
class MeanAPCase:
    name: str
    iou_type: str | tuple[str, ...]
    box_format: str = "xyxy"
    class_metrics: bool = True
    average: str = "macro"
    extended_summary: bool = False
    iou_thresholds: list[float] | None = None
    rec_thresholds: list[float] | None = None
    max_detection_thresholds: list[int] | None = None
    num_images: int = 12
    num_classes: int = 6
    gts_per_image: int = 6
    preds_per_image: int = 12
    seed: int = 123
    height: int = 48
    width: int = 48


def case_benchmark_bbox_xyxy() -> MeanAPCase:
    return MeanAPCase(
        name="benchmark_bbox_xyxy",
        iou_type="bbox",
        box_format="xyxy",
        class_metrics=False,
        num_images=40,
        num_classes=8,
        gts_per_image=10,
        preds_per_image=20,
        seed=321,
    )


def case_benchmark_bbox_class_metrics() -> MeanAPCase:
    return MeanAPCase(
        name="benchmark_bbox_class_metrics",
        iou_type="bbox",
        box_format="xyxy",
        class_metrics=True,
        num_images=30,
        num_classes=8,
        gts_per_image=8,
        preds_per_image=16,
        seed=777,
    )


def case_benchmark_segm_complex() -> MeanAPCase:
    return MeanAPCase(
        name="benchmark_segm_complex",
        iou_type="segm",
        box_format="xyxy",
        class_metrics=True,
        average="micro",
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
        max_detection_thresholds=[1, 10, 100],
        num_images=36,
        num_classes=8,
        gts_per_image=12,
        preds_per_image=24,
        seed=2026,
        height=96,
        width=96,
    )


def case_benchmark_mixed_complex_stable() -> MeanAPCase:
    return MeanAPCase(
        name="benchmark_mixed_complex_stable",
        iou_type=("bbox", "segm"),
        box_format="xyxy",
        class_metrics=False,
        average="micro",
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
        max_detection_thresholds=[1, 10, 100],
        num_images=24,
        num_classes=8,
        gts_per_image=10,
        preds_per_image=20,
        seed=1337,
        height=96,
        width=96,
    )


def case_compare_bbox_xyxy_macro() -> MeanAPCase:
    return MeanAPCase(
        name="compare_bbox_xyxy_macro",
        iou_type="bbox",
        box_format="xyxy",
        class_metrics=True,
        average="macro",
        iou_thresholds=[0.5],
        rec_thresholds=[0.0, 0.5, 1.0],
        max_detection_thresholds=[1, 10, 100],
        num_images=20,
        num_classes=10,
        gts_per_image=8,
        preds_per_image=16,
        seed=999,
    )


def case_compare_bbox_xywh_micro() -> MeanAPCase:
    return MeanAPCase(
        name="compare_bbox_xywh_micro",
        iou_type="bbox",
        box_format="xywh",
        class_metrics=True,
        average="micro",
        iou_thresholds=[0.5],
        rec_thresholds=[0.0, 0.5, 1.0],
        max_detection_thresholds=[1, 10, 100],
        num_images=20,
        num_classes=10,
        gts_per_image=8,
        preds_per_image=16,
        seed=1001,
    )


def case_compare_bbox_cxcywh_macro() -> MeanAPCase:
    return MeanAPCase(
        name="compare_bbox_cxcywh_macro",
        iou_type="bbox",
        box_format="cxcywh",
        class_metrics=True,
        average="macro",
        iou_thresholds=[0.5],
        rec_thresholds=[0.0, 0.5, 1.0],
        max_detection_thresholds=[1, 10, 100],
        num_images=20,
        num_classes=10,
        gts_per_image=8,
        preds_per_image=16,
        seed=1003,
    )


def case_compare_segm_micro() -> MeanAPCase:
    return MeanAPCase(
        name="compare_segm_micro",
        iou_type="segm",
        box_format="xyxy",
        class_metrics=True,
        average="micro",
        iou_thresholds=[0.5],
        rec_thresholds=[0.0, 0.5, 1.0],
        max_detection_thresholds=[1, 10, 100],
        num_images=20,
        num_classes=8,
        gts_per_image=6,
        preds_per_image=12,
        seed=2027,
    )


def case_compare_mixed_stable() -> MeanAPCase:
    return MeanAPCase(
        name="compare_mixed_stable",
        iou_type=("bbox", "segm"),
        box_format="xyxy",
        class_metrics=False,
        average="micro",
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=[0.0, 0.5, 1.0],
        max_detection_thresholds=[1, 10, 50],
        num_images=18,
        num_classes=8,
        gts_per_image=7,
        preds_per_image=14,
        seed=2028,
    )


def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    boxes = torch.zeros((masks.shape[0], 4), dtype=torch.float32)
    for i, mask in enumerate(masks):
        ys, xs = torch.where(mask)
        if xs.numel() == 0 or ys.numel() == 0:
            boxes[i] = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
            continue
        x1 = int(xs.min().item())
        y1 = int(ys.min().item())
        x2 = int(xs.max().item()) + 1
        y2 = int(ys.max().item()) + 1
        boxes[i] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
    return boxes


def _xyxy_to_box_format(boxes_xyxy: torch.Tensor, box_format: str) -> torch.Tensor:
    if box_format == "xyxy":
        return boxes_xyxy

    wh = (boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]).clamp(min=1.0)
    if box_format == "xywh":
        return torch.cat([boxes_xyxy[:, :2], wh], dim=1)
    if box_format == "cxcywh":
        center = boxes_xyxy[:, :2] + 0.5 * wh
        return torch.cat([center, wh], dim=1)

    raise ValueError(f"Unsupported box_format: {box_format}")


def _ensure_valid_xyxy(boxes_xyxy: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
    boxes = boxes_xyxy.clone().to(dtype=torch.float32)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(width - 1))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(height - 1))
    x1y1 = torch.min(boxes[:, :2], boxes[:, 2:])
    x2y2 = torch.max(boxes[:, :2], boxes[:, 2:])
    x2y2[:, 0] = torch.maximum(x2y2[:, 0], x1y1[:, 0] + 1.0)
    x2y2[:, 1] = torch.maximum(x2y2[:, 1], x1y1[:, 1] + 1.0)
    x2y2[:, 0] = x2y2[:, 0].clamp(max=float(width))
    x2y2[:, 1] = x2y2[:, 1].clamp(max=float(height))
    return torch.cat([x1y1, x2y2], dim=1)


def _make_masks(
    *,
    num_objs: int,
    height: int,
    width: int,
    generator: torch.Generator,
) -> torch.Tensor:
    masks = torch.zeros((num_objs, height, width), dtype=torch.bool)
    for j in range(num_objs):
        x1 = int(torch.randint(0, max(1, width - 8), (1,), generator=generator).item())
        y1 = int(torch.randint(0, max(1, height - 8), (1,), generator=generator).item())
        w = int(torch.randint(4, 14, (1,), generator=generator).item())
        h = int(torch.randint(4, 14, (1,), generator=generator).item())
        x2 = min(width, x1 + w)
        y2 = min(height, y1 + h)
        masks[j, y1:y2, x1:x2] = True
    return masks


def make_case_data(
    case: MeanAPCase, *, device: str
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    g = torch.Generator(device="cpu")
    g.manual_seed(case.seed)

    preds: list[dict[str, torch.Tensor]] = []
    target: list[dict[str, torch.Tensor]] = []

    iou_types = case.iou_type if isinstance(case.iou_type, tuple) else (case.iou_type,)
    use_bbox = "bbox" in iou_types
    use_segm = "segm" in iou_types

    for img_idx in range(case.num_images):
        labels_t = torch.randint(0, case.num_classes, (case.gts_per_image,), generator=g, dtype=torch.int64)
        if img_idx == 0:
            n = min(case.gts_per_image, case.num_classes)
            labels_t[:n] = torch.arange(n, dtype=torch.int64)

        masks_t = _make_masks(num_objs=case.gts_per_image, height=case.height, width=case.width, generator=g)
        boxes_t_xyxy = _ensure_valid_xyxy(_masks_to_boxes(masks_t), width=case.width, height=case.height)
        boxes_t = _xyxy_to_box_format(boxes_t_xyxy, case.box_format)

        target_item: dict[str, torch.Tensor] = {
            "labels": labels_t.to(device),
            "iscrowd": torch.zeros((case.gts_per_image,), dtype=torch.int32, device=device),
        }
        if use_bbox:
            target_item["boxes"] = boxes_t.to(device)
        if use_segm:
            target_item["masks"] = masks_t.to(device)
        target.append(target_item)

        src_idx = torch.randint(0, case.gts_per_image, (case.preds_per_image,), generator=g)
        masks_p = masks_t[src_idx].clone()
        for j in range(case.preds_per_image):
            if bool(torch.randint(0, 2, (1,), generator=g).item()):
                masks_p[j] = torch.roll(masks_p[j], shifts=(1, 0), dims=(0, 1))
            if bool(torch.randint(0, 2, (1,), generator=g).item()):
                masks_p[j] = torch.roll(masks_p[j], shifts=(0, 1), dims=(0, 1))

        boxes_p_xyxy = _ensure_valid_xyxy(_masks_to_boxes(masks_p), width=case.width, height=case.height)
        boxes_p = _xyxy_to_box_format(boxes_p_xyxy, case.box_format)
        labels_p = torch.randint(0, case.num_classes, (case.preds_per_image,), generator=g, dtype=torch.int64)
        scores_p = torch.rand((case.preds_per_image,), generator=g, dtype=torch.float32)

        pred_item: dict[str, torch.Tensor] = {
            "labels": labels_p.to(device),
            "scores": scores_p.to(device),
        }
        if use_bbox:
            pred_item["boxes"] = boxes_p.to(device)
        if use_segm:
            pred_item["masks"] = masks_p.to(device)
        preds.append(pred_item)

    return preds, target


def make_metric_kwargs(case: MeanAPCase, *, for_torchmetrics: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "iou_type": case.iou_type,
        "box_format": case.box_format,
        "class_metrics": case.class_metrics,
        "average": case.average,
    }
    if case.extended_summary:
        kwargs["extended_summary"] = case.extended_summary
    if case.iou_thresholds is not None:
        kwargs["iou_thresholds"] = case.iou_thresholds
    if case.rec_thresholds is not None:
        kwargs["rec_thresholds"] = case.rec_thresholds
    if case.max_detection_thresholds is not None:
        kwargs["max_detection_thresholds"] = case.max_detection_thresholds
    if for_torchmetrics:
        kwargs["backend"] = "pycocotools"
    return kwargs


def _assert_single_prefix(
    out_pt: dict[str, torch.Tensor],
    out_tm: dict[str, torch.Tensor],
    *,
    prefix: str,
    max_dets: tuple[int, int, int],
    class_metrics: bool,
) -> None:
    mdt1, mdt2, mdt3 = max_dets
    scalar_keys = [
        f"{prefix}map",
        f"{prefix}map_50",
        f"{prefix}map_75",
        f"{prefix}map_small",
        f"{prefix}map_medium",
        f"{prefix}map_large",
        f"{prefix}mar_{mdt1}",
        f"{prefix}mar_{mdt2}",
        f"{prefix}mar_{mdt3}",
        f"{prefix}mar_small",
        f"{prefix}mar_medium",
        f"{prefix}mar_large",
    ]
    for key in scalar_keys:
        torch.testing.assert_close(out_pt[key].cpu(), out_tm[key].cpu(), rtol=0, atol=1e-4)

    if not class_metrics:
        return

    map_pc_key = f"{prefix}map_per_class"
    mar_pc_key = f"{prefix}mar_{mdt3}_per_class"
    pt_map_pc = out_pt[map_pc_key].cpu().to(dtype=torch.float32)
    tm_map_pc = out_tm[map_pc_key].cpu().to(dtype=torch.float32)
    pt_mar_pc = out_pt[mar_pc_key].cpu().to(dtype=torch.float32)
    tm_mar_pc = out_tm[mar_pc_key].cpu().to(dtype=torch.float32)

    if tm_map_pc.numel() == 1 and tm_map_pc.item() == -1:
        assert pt_map_pc.numel() == 1 and pt_map_pc.item() == -1
        assert pt_mar_pc.numel() == 1 and pt_mar_pc.item() == -1
    else:
        cls_tm = out_tm["classes"].cpu().to(dtype=torch.int64)
        cls_pt = out_pt["classes"].cpu().to(dtype=torch.int64)
        assert torch.equal(torch.sort(cls_tm).values, torch.sort(cls_pt).values)
        order_tm = torch.argsort(cls_tm)
        order_pt = torch.argsort(cls_pt)
        torch.testing.assert_close(pt_map_pc[order_pt], tm_map_pc[order_tm], rtol=0, atol=1e-4)
        torch.testing.assert_close(pt_mar_pc[order_pt], tm_mar_pc[order_tm], rtol=0, atol=1e-4)


def assert_outputs_match(
    out_pt: dict[str, torch.Tensor],
    out_tm: dict[str, torch.Tensor],
    *,
    class_metrics: bool,
    iou_type: str | tuple[str, ...],
    max_detection_thresholds: list[int] | None,
) -> None:
    max_dets_list = max_detection_thresholds or [1, 10, 100]
    assert len(max_dets_list) == 3
    max_dets: tuple[int, int, int] = (int(max_dets_list[0]), int(max_dets_list[1]), int(max_dets_list[2]))

    iou_types = iou_type if isinstance(iou_type, tuple) else (iou_type,)
    if len(iou_types) == 1:
        _assert_single_prefix(out_pt, out_tm, prefix="", max_dets=max_dets, class_metrics=class_metrics)
    else:
        for iou in iou_types:
            _assert_single_prefix(out_pt, out_tm, prefix=f"{iou}_", max_dets=max_dets, class_metrics=class_metrics)
