import pytest
import torch


def _import_metrics():
    pytest.importorskip("torchmetrics")
    pytest.importorskip("lightning_utilities")
    from torchmetrics.detection.mean_ap import MeanAveragePrecision as TMMeanAP

    from pytorchcocotools.lightning.metrics.mean_ap import MeanAveragePrecision as PTMeanAP

    return TMMeanAP, PTMeanAP


def _make_synthetic_bbox_data(
    *,
    num_images: int = 32,
    num_classes: int = 12,
    gts_per_image: int = 8,
    preds_per_image: int = 16,
    seed: int = 123,
    device: str = "cpu",
):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    preds = []
    target = []

    for img_idx in range(num_images):
        # --- targets ---
        labels_t = torch.randint(0, num_classes, (gts_per_image,), generator=g, dtype=torch.int64)
        # Ensure every class appears at least once across the dataset.
        if img_idx == 0:
            n = min(gts_per_image, num_classes)
            labels_t[:n] = torch.arange(n, dtype=torch.int64)

        # generate valid xyxy boxes in a small canvas
        xy1 = torch.rand((gts_per_image, 2), generator=g) * 40
        wh = torch.rand((gts_per_image, 2), generator=g) * 20 + 1
        xy2 = xy1 + wh
        boxes_t = torch.cat([xy1, xy2], dim=1).to(dtype=torch.float32)

        target.append(
            {
                "boxes": boxes_t.to(device),
                "labels": labels_t.to(device),
                "iscrowd": torch.zeros((gts_per_image,), dtype=torch.int32, device=device),
            }
        )

        # --- preds ---
        # base predictions near GT plus some noise
        xy1p = xy1.repeat((preds_per_image // gts_per_image + 1, 1))[:preds_per_image]
        whp = wh.repeat((preds_per_image // gts_per_image + 1, 1))[:preds_per_image]
        noise = (torch.rand((preds_per_image, 4), generator=g) - 0.5) * 5
        boxes_p = torch.cat([xy1p, xy1p + whp], dim=1).to(dtype=torch.float32) + noise
        # fix ordering
        x1y1 = torch.min(boxes_p[:, :2], boxes_p[:, 2:])
        x2y2 = torch.max(boxes_p[:, :2], boxes_p[:, 2:])
        boxes_p = torch.cat([x1y1, x2y2], dim=1)

        labels_p = torch.randint(0, num_classes, (preds_per_image,), generator=g, dtype=torch.int64)
        scores_p = torch.rand((preds_per_image,), generator=g).to(dtype=torch.float32)

        preds.append(
            {
                "boxes": boxes_p.to(device),
                "labels": labels_p.to(device),
                "scores": scores_p.to(device),
            }
        )

    return preds, target


def _make_synthetic_segm_data(
    *,
    num_images: int = 24,
    num_classes: int = 8,
    gts_per_image: int = 6,
    preds_per_image: int = 12,
    height: int = 32,
    width: int = 32,
    seed: int = 123,
    device: str = "cpu",
):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    preds = []
    target = []

    for img_idx in range(num_images):
        # --- targets ---
        labels_t = torch.randint(0, num_classes, (gts_per_image,), generator=g, dtype=torch.int64)
        if img_idx == 0:
            n = min(gts_per_image, num_classes)
            labels_t[:n] = torch.arange(n, dtype=torch.int64)

        masks_t = torch.zeros((gts_per_image, height, width), dtype=torch.bool)
        for j in range(gts_per_image):
            x1 = int(torch.randint(0, max(1, width - 8), (1,), generator=g).item())
            y1 = int(torch.randint(0, max(1, height - 8), (1,), generator=g).item())
            w = int(torch.randint(4, 12, (1,), generator=g).item())
            h = int(torch.randint(4, 12, (1,), generator=g).item())
            x2 = min(width, x1 + w)
            y2 = min(height, y1 + h)
            masks_t[j, y1:y2, x1:x2] = True

        target.append(
            {
                "masks": masks_t.to(device),
                "labels": labels_t.to(device),
                "iscrowd": torch.zeros((gts_per_image,), dtype=torch.int32, device=device),
            }
        )

        # --- preds ---
        src_idx = torch.randint(0, gts_per_image, (preds_per_image,), generator=g)
        masks_p = masks_t[src_idx].clone()
        # light random flips/noise to avoid perfect overlap
        for j in range(preds_per_image):
            if bool(torch.randint(0, 2, (1,), generator=g).item()):
                masks_p[j] = torch.roll(masks_p[j], shifts=(1, 0), dims=(0, 1))
            if bool(torch.randint(0, 2, (1,), generator=g).item()):
                masks_p[j] = torch.roll(masks_p[j], shifts=(0, 1), dims=(0, 1))

        labels_p = torch.randint(0, num_classes, (preds_per_image,), generator=g, dtype=torch.int64)
        scores_p = torch.rand((preds_per_image,), generator=g).to(dtype=torch.float32)

        preds.append(
            {
                "masks": masks_p.to(device),
                "labels": labels_p.to(device),
                "scores": scores_p.to(device),
            }
        )

    return preds, target


def _assert_outputs_match(
    out_pt: dict[str, torch.Tensor],
    out_tm: dict[str, torch.Tensor],
    *,
    prefix: str = "",
    max_dets: tuple[int, int, int] = (1, 10, 100),
    check_classes: bool = True,
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
    for k in scalar_keys:
        torch.testing.assert_close(out_pt[k].cpu(), out_tm[k].cpu(), rtol=0, atol=1e-4)

    if check_classes:
        cls_tm = out_tm["classes"].cpu().to(dtype=torch.int64)
        cls_pt = out_pt["classes"].cpu().to(dtype=torch.int64)
        assert torch.equal(torch.sort(cls_tm).values, torch.sort(cls_pt).values)

    tm_map_pc = out_tm[f"{prefix}map_per_class"].cpu().to(dtype=torch.float32)
    pt_map_pc = out_pt[f"{prefix}map_per_class"].cpu().to(dtype=torch.float32)
    tm_mar_pc = out_tm[f"{prefix}mar_{mdt3}_per_class"].cpu().to(dtype=torch.float32)
    pt_mar_pc = out_pt[f"{prefix}mar_{mdt3}_per_class"].cpu().to(dtype=torch.float32)

    if tm_map_pc.numel() == 1 and tm_map_pc.item() == -1:
        assert pt_map_pc.numel() == 1 and pt_map_pc.item() == -1
        assert pt_mar_pc.numel() == 1 and pt_mar_pc.item() == -1
    else:
        cls_tm = out_tm["classes"].cpu().to(dtype=torch.int64)
        cls_pt = out_pt["classes"].cpu().to(dtype=torch.int64)
        order_tm = torch.argsort(cls_tm)
        order_pt = torch.argsort(cls_pt)
        torch.testing.assert_close(pt_map_pc[order_pt], tm_map_pc[order_tm], rtol=0, atol=1e-4)
        torch.testing.assert_close(pt_mar_pc[order_pt], tm_mar_pc[order_tm], rtol=0, atol=1e-4)


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


def _make_synthetic_bbox_segm_data(
    *,
    num_images: int = 18,
    num_classes: int = 10,
    gts_per_image: int = 7,
    preds_per_image: int = 14,
    height: int = 36,
    width: int = 36,
    seed: int = 909,
    device: str = "cpu",
):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    preds = []
    target = []

    for img_idx in range(num_images):
        labels_t = torch.randint(0, num_classes, (gts_per_image,), generator=g, dtype=torch.int64)
        if img_idx == 0:
            n = min(gts_per_image, num_classes)
            labels_t[:n] = torch.arange(n, dtype=torch.int64)

        masks_t = torch.zeros((gts_per_image, height, width), dtype=torch.bool)
        for j in range(gts_per_image):
            x1 = int(torch.randint(0, max(1, width - 8), (1,), generator=g).item())
            y1 = int(torch.randint(0, max(1, height - 8), (1,), generator=g).item())
            w = int(torch.randint(4, 12, (1,), generator=g).item())
            h = int(torch.randint(4, 12, (1,), generator=g).item())
            x2 = min(width, x1 + w)
            y2 = min(height, y1 + h)
            masks_t[j, y1:y2, x1:x2] = True

        boxes_t = _masks_to_boxes(masks_t)

        target.append(
            {
                "boxes": boxes_t.to(device),
                "masks": masks_t.to(device),
                "labels": labels_t.to(device),
                "iscrowd": torch.zeros((gts_per_image,), dtype=torch.int32, device=device),
            }
        )

        src_idx = torch.randint(0, gts_per_image, (preds_per_image,), generator=g)
        masks_p = masks_t[src_idx].clone()
        for j in range(preds_per_image):
            if bool(torch.randint(0, 2, (1,), generator=g).item()):
                masks_p[j] = torch.roll(masks_p[j], shifts=(1, 0), dims=(0, 1))
            if bool(torch.randint(0, 2, (1,), generator=g).item()):
                masks_p[j] = torch.roll(masks_p[j], shifts=(0, 1), dims=(0, 1))

        boxes_p = _masks_to_boxes(masks_p)
        labels_p = torch.randint(0, num_classes, (preds_per_image,), generator=g, dtype=torch.int64)
        scores_p = torch.rand((preds_per_image,), generator=g).to(dtype=torch.float32)

        preds.append(
            {
                "boxes": boxes_p.to(device),
                "masks": masks_p.to(device),
                "labels": labels_p.to(device),
                "scores": scores_p.to(device),
            }
        )

    return preds, target


@pytest.mark.parametrize("average", ["macro", "micro"])
def test_mean_ap_matches_torchmetrics_bbox(average: str):
    tm_mean_ap, pt_mean_ap = _import_metrics()
    preds, target = _make_synthetic_bbox_data(seed=999)

    # Use a reduced threshold set for speed + stability.
    iou_thresholds = [0.5]
    rec_thresholds = [0.0, 0.5, 1.0]
    max_dets = [1, 10, 100]

    tm = tm_mean_ap(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_dets,
        class_metrics=True,
        average=average,
        backend="pycocotools",
    )
    pt = pt_mean_ap(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_dets,
        class_metrics=True,
        average=average,
    )

    tm.update(preds, target)
    pt.update(preds, target)

    out_tm = tm.compute()
    out_pt = pt.compute()

    _assert_outputs_match(out_pt, out_tm)


@pytest.mark.parametrize("average", ["macro", "micro"])
def test_mean_ap_matches_torchmetrics_segm(average: str):
    tm_mean_ap, pt_mean_ap = _import_metrics()
    preds, target = _make_synthetic_segm_data(seed=2026)

    iou_thresholds = [0.5]
    rec_thresholds = [0.0, 0.5, 1.0]
    max_dets = [1, 10, 100]

    tm = tm_mean_ap(
        iou_type="segm",
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_dets,
        class_metrics=True,
        average=average,
        backend="pycocotools",
    )
    pt = pt_mean_ap(
        iou_type="segm",
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_dets,
        class_metrics=True,
        average=average,
    )

    tm.update(preds, target)
    pt.update(preds, target)

    out_tm = tm.compute()
    out_pt = pt.compute()

    _assert_outputs_match(out_pt, out_tm)


@pytest.mark.parametrize("average", ["macro", "micro"])
def test_mean_ap_matches_torchmetrics_bbox_multi_update(average: str):
    tm_mean_ap, pt_mean_ap = _import_metrics()
    preds, target = _make_synthetic_bbox_data(seed=4242, num_images=20)

    iou_thresholds = [0.5]
    rec_thresholds = [0.0, 0.5, 1.0]
    max_dets = [1, 10, 100]

    tm = tm_mean_ap(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_dets,
        class_metrics=True,
        average=average,
        backend="pycocotools",
    )
    pt = pt_mean_ap(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_dets,
        class_metrics=True,
        average=average,
    )

    # Feed data in two chunks to validate update() accumulation parity.
    split = len(preds) // 2
    tm.update(preds[:split], target[:split])
    pt.update(preds[:split], target[:split])
    tm.update(preds[split:], target[split:])
    pt.update(preds[split:], target[split:])

    out_tm = tm.compute()
    out_pt = pt.compute()

    _assert_outputs_match(out_pt, out_tm)


@pytest.mark.parametrize("average", ["macro", "micro"])
def test_mean_ap_matches_torchmetrics_bbox_segm_multi_update(average: str):
    tm_mean_ap, pt_mean_ap = _import_metrics()
    preds, target = _make_synthetic_bbox_segm_data(seed=2027)

    iou_thresholds = [0.5, 0.75]
    rec_thresholds = [0.0, 0.5, 1.0]
    max_dets = [1, 10, 50]

    tm = tm_mean_ap(
        box_format="xyxy",
        iou_type=("bbox", "segm"),
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_dets,
        class_metrics=False,
        average=average,
        backend="pycocotools",
    )
    pt = pt_mean_ap(
        box_format="xyxy",
        iou_type=("bbox", "segm"),
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_dets,
        class_metrics=False,
        average=average,
    )

    split = len(preds) // 3
    tm.update(preds[:split], target[:split])
    pt.update(preds[:split], target[:split])
    tm.update(preds[split : 2 * split], target[split : 2 * split])
    pt.update(preds[split : 2 * split], target[split : 2 * split])
    tm.update(preds[2 * split :], target[2 * split :])
    pt.update(preds[2 * split :], target[2 * split :])

    out_tm = tm.compute()
    out_pt = pt.compute()

    _assert_outputs_match(
        out_pt,
        out_tm,
        prefix="bbox_",
        max_dets=(max_dets[0], max_dets[1], max_dets[2]),
        check_classes=True,
    )
    _assert_outputs_match(
        out_pt,
        out_tm,
        prefix="segm_",
        max_dets=(max_dets[0], max_dets[1], max_dets[2]),
        check_classes=False,
    )
