from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import copy
import datetime
import logging
from logging import Logger
import time
from typing import cast

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools import mask
from pytorchcocotools.coco import COCO
from pytorchcocotools.internal.cocoeval_types import EvalImgResult, EvalResult, Params
from pytorchcocotools.internal.entities import IoUObject, IoUType, Range, RangeLabel, RleObj, TorchDevice
from pytorchcocotools.internal.structure.annotations import (
    CocoAnnotationDetection,
    CocoAnnotationKeypointDetection,
    CocoAnnotationObjectDetection,
    Segmentation,
)
from pytorchcocotools.utils.logging import get_logger


class COCOeval:
    logger: Logger

    def __init__(
        self,
        cocoGt: COCO | None = None,  # noqa: N803
        cocoDt: COCO | None = None,  # noqa: N803
        iouType: IoUType = "segm",  # noqa: N803
        *,
        enable_logging: bool = True,
        device: TorchDevice | None = None,
        requires_grad: bool = False,
    ) -> None:
        """Initialize CocoEval using coco APIs for gt and dt.

        Args:
            cocoGt: COCO object with ground truth annotations. Defaults to None.
            cocoDt: COCO object with detection results. Defaults to None.
            iouType: _description_. Defaults to "segm".
            enable_logging: Whether to enable logging. Defaults to True.
            device: The desired device of the bounding boxes.
            requires_grad: Whether the bounding boxes require gradients.
        """
        self.logger = get_logger(__name__) if enable_logging else logging.getLogger(__name__)
        if not iouType:
            self.logger.info("iouType not specified. use default iouType segm")
        self.cocoGt = cocoGt or COCO()  # ground truth COCO API
        self.cocoDt = cocoDt or COCO()  # detections COCO API
        self.eval_imgs: list[EvalImgResult | None] = []  # per-image per-category evaluation results [KxAxI] elements
        self.eval = EvalResult()  # accumulated evaluation results
        self._gts = defaultdict(list[CocoAnnotationDetection])  # gt for evaluation
        self._dts = defaultdict(list[CocoAnnotationDetection])  # dt for evaluation
        self.params: Params = Params(iouType=iouType)  # parameters
        self._paramsEval: Params = Params(iouType=iouType)  # parameters for evaluation
        self.stats: Tensor = torch.Tensor()  # result summarization
        self.ious: dict[tuple[int, int], Tensor] = {}  # ious between all gts and dts
        self.device = device
        self.requires_grad = requires_grad if requires_grad is not None else False
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self) -> None:
        """Prepare ._gts and ._dts for evaluation based on params."""

        def _toMask(anns: list[CocoAnnotationDetection], coco: COCO) -> list[CocoAnnotationDetection]:  # noqa: N802
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                # TODO: fix and validate
                ann.segmentation = cast(list[Segmentation], rle)  # type: ignore  # noqa: PGH003
            return anns

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            gts = _toMask(gts, self.cocoGt)
            dts = _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt.ignore = gt.iscrowd
            if p.iouType == "keypoints":
                # using dictionary access allows mixed datasets (instances and pose)
                gt.ignore = (gt.get("num_keypoints", 0) == 0) or gt.ignore
        self._gts = defaultdict(list[CocoAnnotationDetection])  # gt for evaluation
        self._dts = defaultdict(list[CocoAnnotationDetection])  # dt for evaluation
        for gt in gts:
            self._gts[gt.image_id, gt.category_id].append(gt)
        for dt in dts:
            self._dts[dt.image_id, dt.category_id].append(dt)
        self.eval_imgs: list[EvalImgResult | None] = []  # per-image per-category evaluation results
        self.eval = EvalResult()

    def evaluate(self) -> None:
        """Run per image evaluation on given images and store results (a list of dict) in self.evalImgs."""
        tic = time.time()
        self.logger.info("Running per image evaluation...")
        p = self.params
        self.logger.info(f"Evaluate annotation type *{p.iouType}*")
        p.imgIds = torch.unique(torch.tensor(p.imgIds, device=self.device, requires_grad=self.requires_grad)).tolist()
        if p.useCats:
            p.catIds = torch.unique(
                torch.tensor(p.catIds, device=self.device, requires_grad=self.requires_grad)
            ).tolist()
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        cat_ids = p.catIds if p.useCats else [-1]

        compute_iou: Callable[[int, int], Tensor] = lambda img_id, cat_id: torch.Tensor()  # noqa: E731
        if p.iouType == "segm" or p.iouType == "bbox":
            compute_iou = self.computeIoU
        elif p.iouType == "keypoints":
            compute_iou = self.computeOks
        self.ious = {(img_id, cat_id): compute_iou(img_id, cat_id) for img_id in p.imgIds for cat_id in cat_ids}

        max_det = p.maxDets[-1]
        self.eval_imgs = [
            self.evaluateImg(imgId, catId, areaRng, max_det)
            for catId in cat_ids
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        self.logger.info(f"DONE (t={toc - tic:0.2f}s).")

    def computeIoU(self, imgId: int, catId: int) -> Tensor:  # noqa: N803, N802
        """Compute the IoU between detections and ground truth for the given :param:`imgId` and :param:`catId`.

        Args:
            imgId: The image id.
            catId: The category id.

        Raises:
            ValueError: Unknown iouType for iou computation.

        Returns:
            The IoU between detections and ground truth.
        """
        if self.params.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            # Optimized: Use list.extend instead of nested list comprehension
            gt = []
            dt = []
            for c_id in self.params.catIds:
                gt.extend(self._gts[imgId, c_id])
                dt.extend(self._dts[imgId, c_id])

        gt = cast(list[CocoAnnotationObjectDetection], gt)
        dt = cast(list[CocoAnnotationObjectDetection], dt)
        if len(gt) == 0 or len(dt) == 0:
            return Tensor([])
        inds = torch.argsort(
            torch.tensor(
                [-d.score if d.score is not None else 0 for d in dt],
                device=self.device,
                requires_grad=self.requires_grad,
            )
        )
        # Optimized: Use list comprehension with direct indexing
        dt = [dt[i] for i in inds.tolist()]
        if len(dt) > self.params.maxDets[-1]:
            dt = dt[0 : self.params.maxDets[-1]]

        # TODO: remove
        def to_internal_types(segmentation: Segmentation, size: tuple[int, int]) -> RleObj:
            if isinstance(segmentation, dict):
                return RleObj(counts=segmentation["counts"], size=segmentation["size"])
            if isinstance(segmentation, RleObj):
                return cast(RleObj, segmentation)
            return segmentation  # type: ignore  # noqa: PGH003 # TODO: fix

        #     if isinstance(segmentation, CocoRLE):
        #         return RLE(
        #             segmentation.size[0],
        #             segmentation.size[1],
        #             torch.tensor(segmentation.counts, device=self.device, requires_grad=self.requires_grad),
        #         )
        #     poly = Polygon(
        #         torch.tensor(
        #             segmentation, dtype=torch.float64, device=self.device, requires_grad=self.requires_grad
        #         ).view(-1, 2),
        #         canvas_size=(size[0], size[1]),
        #         device=self.device,
        #         requires_grad=self.requires_grad,
        #     )  # ty:ignore[no-matching-overload]
        #     return rleFrPoly(poly, device=self.device, requires_grad=self.requires_grad)

        img = self.cocoGt.loadImgs(imgId)[0]
        size = img.height, img.width

        if self.params.iouType == "segm":
            g = cast(IoUObject, [to_internal_types(g.segmentation, size) for g in gt])
            d = cast(IoUObject, [to_internal_types(d.segmentation, size) for d in dt])
        elif self.params.iouType == "bbox":
            g = tv.BoundingBoxes(
                torch.tensor(
                    [g.bbox for g in gt], dtype=torch.float32, device=self.device, requires_grad=self.requires_grad
                ),
                format=tv.BoundingBoxFormat.XYWH,
                canvas_size=size,
                device=self.device,
                requires_grad=self.requires_grad,
            )  # ty:ignore[no-matching-overload]
            d = tv.BoundingBoxes(
                torch.tensor(
                    [d.bbox for d in dt], dtype=torch.float32, device=self.device, requires_grad=self.requires_grad
                ),
                format=tv.BoundingBoxFormat.XYWH,
                canvas_size=size,
                device=self.device,
                requires_grad=self.requires_grad,
            )  # ty:ignore[no-matching-overload]
            if g.numel() == 0 or d.numel() == 0:
                return Tensor([])
        else:
            raise ValueError("Unknown iouType for iou computation.")  # noqa: TRY002

        # compute iou between each dt and gt region
        iscrowd = [bool(o.iscrowd) for o in gt]
        ious = mask.iou(d, g, iscrowd)
        return ious.float()  # Normalize to float32; evaluateImg upcasts to float64 internally

    @torch.compile(fullgraph=False, mode="reduce-overhead")
    def computeOks(self, imgId: int, catId: int) -> Tensor:  # noqa: N803, N802
        # dimention here should be Nxm
        gts = cast(list[CocoAnnotationKeypointDetection], self._gts[imgId, catId])
        dts = cast(list[CocoAnnotationKeypointDetection], self._dts[imgId, catId])
        inds = torch.argsort(
            torch.tensor(
                [-d.score if d.score is not None else 0 for d in dts],
                device=self.device,
                requires_grad=self.requires_grad,
            )
        )
        # Optimized: Convert to list once for indexing
        dts = [dts[i] for i in inds.tolist()]
        if len(dts) > self.params.maxDets[-1]:
            dts = dts[0 : self.params.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return torch.tensor([], device=self.device, requires_grad=self.requires_grad)
        ious = torch.zeros((len(dts), len(gts)), device=self.device, requires_grad=self.requires_grad)
        sigmas = self.params.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)

        # Optimized: Vectorize OKS computation
        # Pre-convert all keypoints and bboxes to tensors
        if len(gts) > 0 and len(dts) > 0:
            # Stack all gt keypoints and bboxes
            gt_keypoints = torch.stack(
                [torch.tensor(gt.keypoints, device=self.device, requires_grad=self.requires_grad) for gt in gts]
            )  # [num_gts, 3*k]
            gt_bboxes = torch.stack(
                [torch.tensor(gt.bbox, device=self.device, requires_grad=self.requires_grad) for gt in gts]
            )  # [num_gts, 4]
            gt_areas = torch.tensor([gt.area for gt in gts], device=self.device, requires_grad=self.requires_grad)

            # Stack all dt keypoints
            dt_keypoints = torch.stack(
                [torch.tensor(dt.keypoints, device=self.device, requires_grad=self.requires_grad) for dt in dts]
            )  # [num_dts, 3*k]

            # Extract x, y, v for all gts and dts
            xg = gt_keypoints[:, 0::3]  # [num_gts, k]
            yg = gt_keypoints[:, 1::3]  # [num_gts, k]
            vg = gt_keypoints[:, 2::3]  # [num_gts, k]

            xd = dt_keypoints[:, 0::3]  # [num_dts, k]
            yd = dt_keypoints[:, 1::3]  # [num_dts, k]

            # Compute k1 for all gts
            k1 = torch.count_nonzero(vg > 0, dim=1)  # [num_gts]

            # Compute bounds for all gts
            x0 = gt_bboxes[:, 0] - gt_bboxes[:, 2]  # [num_gts]
            x1 = gt_bboxes[:, 0] + gt_bboxes[:, 2] * 2  # [num_gts]
            y0 = gt_bboxes[:, 1] - gt_bboxes[:, 3]  # [num_gts]
            y1 = gt_bboxes[:, 1] + gt_bboxes[:, 3] * 2  # [num_gts]

            # Compute distances for all dt-gt pairs
            # dx, dy shape: [num_dts, num_gts, k]
            dx = xd.unsqueeze(1) - xg.unsqueeze(0)  # [num_dts, num_gts, k]
            dy = yd.unsqueeze(1) - yg.unsqueeze(0)  # [num_dts, num_gts, k]

            # Vectorised replacement for the k1==0 bounds inner loop.
            k1_zero = k1 == 0  # [G]
            if k1_zero.any():
                # dx/dy bounds for k1==0 GTs: clamp(x0[j]-xd[i], 0) + clamp(xd[i]-x1[j], 0).
                # Shapes after broadcasting: [D, G, k].
                lower_x = (x0[None, :, None] - xd[:, None, :]).clamp(min=0)
                upper_x = (xd[:, None, :] - x1[None, :, None]).clamp(min=0)
                lower_y = (y0[None, :, None] - yd[:, None, :]).clamp(min=0)
                upper_y = (yd[:, None, :] - y1[None, :, None]).clamp(min=0)
                dx_bounds = lower_x + upper_x  # [D, G, k]
                dy_bounds = lower_y + upper_y  # [D, G, k]
                dx = torch.where(k1_zero[None, :, None], dx_bounds, dx)
                dy = torch.where(k1_zero[None, :, None], dy_bounds, dy)

            # Compute OKS for all pairs
            e = (
                (torch.pow(dx, 2) + torch.pow(dy, 2))
                / vars.unsqueeze(0).unsqueeze(0)
                / (gt_areas.unsqueeze(0).unsqueeze(-1) + torch.finfo(torch.float32).eps)
                / 2
            )

            # Vectorised visibility masking (replaces per-GT loop).
            # For k1>0 GTs: non-visible keypoints get e=inf so exp(-inf)=0.
            # For k1==0 GTs: all keypoints contribute.
            visible_all = vg > 0  # [G, k]
            k1_pos = k1 > 0  # [G]
            e = torch.where(
                k1_pos[None, :, None] & ~visible_all[None, :, :],
                torch.full_like(e, float("inf")),
                e,
            )  # [D, G, k]
            exp_neg_e = torch.exp(-e).sum(dim=2)  # [D, G]
            # Denominator: number of visible keypoints (k1) or k (for k1==0).
            denom_counts = torch.where(k1_pos, k1.float(), torch.full_like(k1, float(k), dtype=torch.float32))  # [G]
            ious = exp_neg_e / denom_counts[None, :]  # [D, G]

        return ious

    def evaluateImg(self, imgId: int, catId: int, aRng: Range, maxDet: int) -> EvalImgResult | None:  # noqa: N803, N802
        """Perform evaluation for single category and image.

        Args:
            imgId: The image id.
            catId: The category id.
            aRng: The area range.
            maxDet: The maximum number of detections.

        Returns:
            The image evaluation result.
        """
        if self.params.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            # Optimized: Use list.extend instead of nested list comprehension
            gt = []
            dt = []
            for c_id in self.params.catIds:
                gt.extend(self._gts[imgId, c_id])
                dt.extend(self._dts[imgId, c_id])
        if len(gt) == 0 and len(dt) == 0:
            return None

        ignored = [1 if g.ignore or (g.area < aRng[0] or g.area > aRng[1]) else 0 for g in gt]

        gt_ig = torch.tensor(ignored, device=self.device, requires_grad=self.requires_grad)
        # sort dt highest score first, sort gt ignore last
        gtind = torch.argsort(gt_ig, stable=True)
        gt_ig = gt_ig[gtind]
        # Optimized: Convert to list once for indexing
        gt = [gt[i] for i in gtind.tolist()]
        dtind = torch.argsort(
            torch.tensor(
                [-d.score if d.score is not None else 0 for d in dt],
                device=self.device,
                requires_grad=self.requires_grad,
            ),
            stable=True,
        )
        # Optimized: Convert to list once for indexing
        dt = [dt[i] for i in dtind[0:maxDet].tolist()]
        iscrowd_t = torch.tensor([int(o.iscrowd) for o in gt], device=self.device, requires_grad=self.requires_grad)
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        # Precompute clamped IoU thresholds once (avoids tensor alloc per loop iteration).
        _limit = float(1.0 - torch.finfo(torch.float32).eps)
        iou_thrs_t = torch.tensor(
            [min(float(t.item()), _limit) for t in self.params.iouThrs],
            device=self.device,
            dtype=torch.float64,
        )  # [T]

        num_iou_thrs = len(self.params.iouThrs)  # T
        num_gt = len(gt)  # G
        num_dt = len(dt)  # D
        gtm = torch.zeros((num_iou_thrs, num_gt), device=self.device, requires_grad=self.requires_grad)
        dtm = torch.zeros((num_iou_thrs, num_dt), device=self.device, requires_grad=self.requires_grad)
        dt_ig = torch.zeros((num_iou_thrs, num_dt), device=self.device, requires_grad=self.requires_grad)
        if len(ious) != 0:
            # --- Vectorized matching: iterate over D detections, vectorize over T thresholds ---
            # GTs are already sorted: non-ignored (gt_ig==0) first, ignored (gt_ig==1) last.
            iscrowd_bool = iscrowd_t.bool()  # [G]
            gt_ig_bool = gt_ig.bool()  # [G]
            non_ignored = ~gt_ig_bool  # [G]

            # Pre-extract GT and DT ids as tensors for fast assignment
            gt_ids_t = torch.tensor([g.id for g in gt], dtype=dtm.dtype, device=self.device)  # [G]
            dt_ids_t = torch.tensor([d.id for d in dt], dtype=gtm.dtype, device=self.device)  # [D]

            # ious: [D, G] already (subset-selected above)
            ious_f = ious.to(dtype=torch.float64, device=self.device)  # [D, G]

            # matched_gt_t[t, g] = True when GT g has been matched at threshold t
            matched_gt_t = torch.zeros((num_iou_thrs, num_gt), dtype=torch.bool, device=self.device)

            # Greedy matching per detection (sequential over D), vectorized over T:
            for dind in range(num_dt):
                iou_d = ious_f[dind]  # [G] – IoU of this detection against all GTs

                # Expand across thresholds: [T, G]
                iou_dt = iou_d.unsqueeze(0).expand(num_iou_thrs, -1)  # [T, G]
                t_thrs = iou_thrs_t.unsqueeze(1)  # [T, 1]
                above_thr = iou_dt >= t_thrs  # [T, G]

                # GT is available: not matched yet OR is a crowd annotation
                avail = ~matched_gt_t | iscrowd_bool.unsqueeze(0)  # [T, G]

                # --- Non-ignored GT candidates ---
                # Valid: above threshold, available, and non-ignored
                ni_valid = above_thr & avail & non_ignored.unsqueeze(0)  # [T, G]
                ni_score = iou_dt.masked_fill(~ni_valid, -1.0)  # [T, G]
                best_ni_val, best_ni_idx = ni_score.max(dim=1)  # [T]
                has_ni = best_ni_val >= 0  # [T]  (>= threshold because we masked with >=)
                # Note: best_ni_val == -1 means no valid non-ignored match

                # --- Ignored GT candidates (only if no non-ignored match found) ---
                # (Original algorithm: once a non-ignored match is found at threshold t,
                #  we stop before considering ignored GTs → they won't be looked at.)
                ig_valid = above_thr & avail & gt_ig_bool.unsqueeze(0) & ~has_ni.unsqueeze(1)  # [T, G]
                ig_score = iou_dt.masked_fill(~ig_valid, -1.0)  # [T, G]
                best_ig_val, best_ig_idx = ig_score.max(dim=1)  # [T]
                has_ig = best_ig_val >= 0  # [T]

                # Best match per threshold: prefer non-ignored
                best_idx = torch.where(has_ni, best_ni_idx, best_ig_idx)  # [T]
                has_match = has_ni | has_ig  # [T]

                if not has_match.any():
                    continue

                # Clamp for safe indexing (irrelevant values masked by has_match)
                safe_idx = best_idx.clamp(0, num_gt - 1)  # [T]

                # Update dtm[t, dind] = gt[m].id  for matched thresholds
                dtm[:, dind] = torch.where(has_match, gt_ids_t[safe_idx], dtm[:, dind])

                # Update dt_ig[t, dind] = gt_ig[m]  for matched thresholds
                dt_ig[:, dind] = torch.where(has_match, gt_ig_bool[safe_idx].to(dt_ig.dtype), dt_ig[:, dind])

                # Update gtm[t, m] = dt[dind].id  — scatter to different g per t
                t_indices = torch.where(has_match)[0]  # which thresholds matched
                g_indices = safe_idx[t_indices]  # which GT they matched
                gtm[t_indices, g_indices] = dt_ids_t[dind]

                # Mark matched GTs (not crowds): matched_gt_t[t, m] = True
                is_crowd_match = iscrowd_bool[safe_idx]  # [T]
                should_mark = has_match & ~is_crowd_match
                t_mark = torch.where(should_mark)[0]
                g_mark = safe_idx[t_mark]
                matched_gt_t[t_mark, g_mark] = True

        # set unmatched detections outside of area range to ignore
        a = torch.tensor(
            [d.area < aRng[0] or d.area > aRng[1] for d in dt], device=self.device, requires_grad=self.requires_grad
        ).reshape((1, len(dt)))
        dt_ig = torch.logical_or(
            dt_ig, torch.logical_and(dtm == 0, torch.repeat_interleave(a, repeats=num_iou_thrs, dim=0))
        )
        # store results for given image and category
        return EvalImgResult(
            image_id=imgId,
            category_id=catId,
            aRng=aRng,
            maxDet=maxDet,
            dtIds=torch.tensor([d.id for d in dt], device=self.device, requires_grad=self.requires_grad),
            gtIds=torch.tensor([g.id for g in gt], device=self.device, requires_grad=self.requires_grad),
            dtMatches=dtm,
            gtMatches=gtm,
            dtScores=torch.tensor([d.score for d in dt], device=self.device, requires_grad=self.requires_grad),
            gtIgnore=gt_ig,
            dtIgnore=dt_ig,
        )

    def accumulate(self, p: Params | None = None) -> None:
        """Accumulate per image evaluation results and store the result in self.eval.

        Args:
            p: Input params for evaluation. Defaults to None.
        """
        self.logger.info("Accumulating evaluation results...")
        tic = time.time()
        if not self.eval_imgs or len(self.eval_imgs) == 0:
            self.logger.info("Please run evaluate() first")
        # allows input customized parameters
        p = p or self.params

        ############
        #
        # K = category ids
        # A = area range indices
        # M = max dets
        # I = image ids
        # T = iou thresholds
        # R = recall thresholds
        #
        ############

        p.catIds = p.catIds if p.useCats == 1 else [-1]
        num_iou_thrs = len(p.iouThrs)  # T
        num_rec_thrs = len(p.recThrs)  # R
        num_cat_ids = len(p.catIds) if p.useCats else 1  # K
        num_area_rng = len(p.areaRng)  # A
        num_max_dets = len(p.maxDets)  # M
        precision = -torch.ones(
            (num_iou_thrs, num_rec_thrs, num_cat_ids, num_area_rng, num_max_dets),
            device=self.device,
            dtype=torch.float64,
            requires_grad=self.requires_grad,
        )  # -1 for the precision of absent categories
        recall = -torch.ones(
            (num_iou_thrs, num_cat_ids, num_area_rng, num_max_dets),
            device=self.device,
            dtype=torch.float64,
            requires_grad=self.requires_grad,
        )
        scores = -torch.ones(
            (num_iou_thrs, num_rec_thrs, num_cat_ids, num_area_rng, num_max_dets),
            device=self.device,
            dtype=torch.float64,
            requires_grad=self.requires_grad,
        )

        # create dictionary for future indexing
        cat_ids = self._paramsEval.catIds if self._paramsEval.useCats else [-1]
        set_cat_ids = set(cat_ids)
        set_area_rng = set(map(tuple, self._paramsEval.areaRng))
        set_max_dets = set(self._paramsEval.maxDets)
        set_img_ids = set(self._paramsEval.imgIds)
        # get inds to evaluate
        cat_ids_indices = [n for n, k in enumerate(p.catIds) if k in set_cat_ids]
        max_dets_indices = [m for n, m in enumerate(p.maxDets) if m in set_max_dets]
        area_rng_indices = [n for n, a in enumerate(tuple(x) for x in p.areaRng) if a in set_area_rng]
        img_ids_indices = [n for n, i in enumerate(p.imgIds) if i in set_img_ids]
        len_img_ids = len(self._paramsEval.imgIds)  # I0
        len_area_rng = len(self._paramsEval.areaRng)  # A0
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(cat_ids_indices):
            num_k = k0 * len_area_rng * len_img_ids
            for a, a0 in enumerate(area_rng_indices):
                num_a = a0 * len_img_ids
                for m, max_det in enumerate(max_dets_indices):
                    eval_img_results = [self.eval_imgs[num_k + num_a + i] for i in img_ids_indices]  # E
                    eval_img_results = [el for el in eval_img_results if el is not None]
                    if len(eval_img_results) == 0:
                        continue
                    dt_scores = torch.concatenate([el.dtScores[0:max_det] for el in eval_img_results]).to(
                        dtype=torch.float64
                    )

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = torch.argsort(dt_scores, descending=True, stable=True)
                    dt_scores_sorted = dt_scores[inds]

                    dt_matches = torch.cat([el.dtMatches[:, 0:max_det] for el in eval_img_results], dim=1)[:, inds]
                    dt_ig = torch.cat([el.dtIgnore[:, 0:max_det] for el in eval_img_results], dim=1)[:, inds]
                    gt_ig = torch.cat([el.gtIgnore for el in eval_img_results])
                    npig = torch.count_nonzero(gt_ig == 0)
                    if npig == 0:
                        continue
                    tps = torch.logical_and(dt_matches, torch.logical_not(dt_ig))
                    fps = torch.logical_and(torch.logical_not(dt_matches), torch.logical_not(dt_ig))

                    tp_sum = torch.cumsum(tps, dim=1, dtype=torch.float64)
                    fp_sum = torch.cumsum(fps, dim=1, dtype=torch.float64)
                    # Vectorized over all T IoU thresholds at once:
                    nd = tp_sum.shape[1]
                    if nd == 0:
                        # No detections: set recall/precision/scores to 0 for all thresholds.
                        recall[:, k, a, m] = 0.0
                        precision[:, :, k, a, m] = 0.0
                        scores[:, :, k, a, m] = 0.0
                        continue
                    eps64 = torch.finfo(torch.float64).eps
                    # rc_all, pr_all: [T, nd]
                    rc_all = tp_sum / npig
                    pr_all = tp_sum / (fp_sum + tp_sum + eps64)
                    # Backward cummax: each position becomes max(itself, all positions to its right).
                    # Equivalent to the original Python loop: for i in range(nd-1,0,-1): pr[i-1]=max(pr[i-1],pr[i])
                    pr_all = torch.flip(torch.cummax(torch.flip(pr_all, dims=[1]), dim=1).values, dims=[1])
                    recall[:, k, a, m] = rc_all[:, -1]
                    # Vectorized searchsorted across all T thresholds: [T, R]
                    rec_thrs_2d = (
                        p.recThrs.to(device=rc_all.device, dtype=rc_all.dtype)
                        .unsqueeze(0)
                        .expand(num_iou_thrs, -1)
                        .contiguous()
                    )
                    rec_inds = torch.searchsorted(rc_all, rec_thrs_2d, side="left")  # [T, R]
                    valid = rec_inds < nd  # always >= 0 for side="left"
                    rec_inds_c = rec_inds.clamp(0, nd - 1)
                    # Gather precision and score values for valid positions
                    q_all = pr_all.gather(1, rec_inds_c)  # [T, R]
                    q_all[~valid] = 0.0
                    ss_all = dt_scores_sorted.unsqueeze(0).expand(num_iou_thrs, nd).gather(1, rec_inds_c)  # [T, R]
                    ss_all[~valid] = 0.0
                    precision[:, :, k, a, m] = q_all
                    scores[:, :, k, a, m] = ss_all
        self.eval = EvalResult(
            params=p,
            counts=[num_iou_thrs, num_rec_thrs, num_cat_ids, num_area_rng, num_max_dets],
            date=datetime.datetime.now(),  # .strftime("%Y-%m-%d %H:%M:%S"),
            precision=precision.to(dtype=torch.float32),
            recall=recall.to(dtype=torch.float32),
            scores=scores.to(dtype=torch.float32),
        )
        toc = time.time()
        self.logger.info(f"DONE (t={toc - tic:0.2f}s).")

    def summarize(self) -> None:
        """Compute and display summary metrics for evaluation results.

        Note:
            This function can *only* be applied on the default parameter setting.
        """

        def _summarize(
            ap: int | bool = True,
            iouThr: float | None = None,  # noqa: N803
            areaRng: RangeLabel = "all",  # noqa: N803
            maxDets: int = 100,  # noqa: N803
        ) -> Tensor:
            ap = bool(ap)
            template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            title = "Average Precision" if ap else "Average Recall"
            type_abbrv = "(AP)" if ap else "(AR)"
            iou = (
                f"{self.params.iouThrs[0].item():0.2f}:{self.params.iouThrs[-1].item():0.2f}"
                if iouThr is None
                else f"{iouThr:0.2f}"
            )

            aind = [i for i, aRng in enumerate(self.params.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(self.params.maxDets) if mDet == maxDets]
            if ap:
                # dimension of precision: [TxRxKxAxM]
                prec = self.eval.precision
                # IoU
                if iouThr is not None:
                    iou_thr_tensor = torch.tensor(iouThr, device=self.device, requires_grad=self.requires_grad)
                    thr = torch.where(iou_thr_tensor == self.params.iouThrs)[0]
                    prec = prec[thr]
                prec = prec[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                prec = self.eval.recall
                if iouThr is not None:
                    iou_thr_tensor = torch.tensor(iouThr, device=self.device, requires_grad=self.requires_grad)
                    thr = torch.where(iou_thr_tensor == self.params.iouThrs)[0]
                    prec = prec[thr]
                prec = prec[:, :, aind, mind]
            mean_s = (
                torch.tensor([-1.0], device=self.device, requires_grad=self.requires_grad)
                if len(prec[prec > -1]) == 0
                else torch.mean(prec[prec > -1])
            )
            self.logger.info(template.format(title, type_abbrv, iou, areaRng, maxDets, mean_s.item()))
            return mean_s

        def _summarizeDets() -> Tensor:  # noqa: N802
            stats = torch.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps() -> Tensor:  # noqa: N802
            stats = torch.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")  # noqa: TRY002
        iou_type = self.params.iouType

        summarize: Callable[[], Tensor] = lambda: torch.Tensor()  # noqa: E731

        if iou_type == "segm" or iou_type == "bbox":
            summarize = _summarizeDets
        elif iou_type == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self) -> str:
        self.summarize()
        return str(self.stats)  # should return a string but just calculates it
