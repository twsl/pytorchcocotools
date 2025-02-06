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
        requires_grad: bool | None = None,
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
            gt = [_ for c_id in self.params.catIds for _ in self._gts[imgId, c_id]]
            dt = [_ for c_id in self.params.catIds for _ in self._dts[imgId, c_id]]

        gt = cast(list[CocoAnnotationObjectDetection], gt)
        dt = cast(list[CocoAnnotationObjectDetection], dt)
        if len(gt) == 0 and len(dt) == 0:
            return Tensor([])
        inds = torch.argsort(
            torch.tensor(
                [-d.score if d.score is not None else 0 for d in dt],
                device=self.device,
                requires_grad=self.requires_grad,
            )
        )
        dt = [dt[i] for i in inds]  # TODO: optimize, dt[inds]
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
        #     )  # pyright: ignore[reportCallIssue]
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
            )  # pyright: ignore[reportCallIssue]
            d = tv.BoundingBoxes(
                torch.tensor(
                    [d.bbox for d in dt], dtype=torch.float32, device=self.device, requires_grad=self.requires_grad
                ),
                format=tv.BoundingBoxFormat.XYWH,
                canvas_size=size,
                device=self.device,
                requires_grad=self.requires_grad,
            )  # pyright: ignore[reportCallIssue]
        else:
            raise ValueError("Unknown iouType for iou computation.")  # noqa: TRY002

        # compute iou between each dt and gt region
        iscrowd = [bool(o.iscrowd) for o in gt]
        ious = mask.iou(d, g, iscrowd)
        return ious

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
        dts = [dts[i] for i in inds]
        if len(dts) > self.params.maxDets[-1]:
            dts = dts[0 : self.params.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return torch.tensor([], device=self.device, requires_grad=self.requires_grad)
        ious = torch.zeros((len(dts), len(gts)), device=self.device, requires_grad=self.requires_grad)
        sigmas = self.params.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = torch.tensor(gt.keypoints, device=self.device, requires_grad=self.requires_grad)
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = torch.count_nonzero(vg > 0)
            bb = torch.tensor(gt.bbox, device=self.device, requires_grad=self.requires_grad)
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = torch.tensor(dt.keypoints, device=self.device, requires_grad=self.requires_grad)
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = torch.zeros(k)
                    dx, _ = torch.max(torch.stack([z, x0 - xd]), dim=0) + torch.max(torch.stack([z, xd - x1]), dim=0)
                    dy, _ = torch.max(torch.stack([z, y0 - yd]), dim=0) + torch.max(torch.stack([z, yd - y1]), dim=0)
                e = (torch.pow(dx, 2) + torch.pow(dy, 2)) / vars / (gt.area + torch.finfo(torch.float32).eps) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = torch.sum(torch.exp(-e)) / e.shape[0]
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
            gt = [_ for c_id in self.params.catIds for _ in self._gts[imgId, c_id]]
            dt = [_ for c_id in self.params.catIds for _ in self._dts[imgId, c_id]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        ignored = [1 if g.ignore or (g.area < aRng[0] or g.area > aRng[1]) else 0 for g in gt]

        gt_ig = torch.Tensor(ignored)
        # sort dt highest score first, sort gt ignore last
        gtind = torch.argsort(gt_ig)
        gt = [gt[i] for i in gtind]
        dtind = torch.argsort(
            torch.tensor(
                [-d.score if d.score is not None else 0 for d in dt],
                device=self.device,
                requires_grad=self.requires_grad,
            )
        )
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = torch.tensor([int(o.iscrowd) for o in gt], device=self.device, requires_grad=self.requires_grad)
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        num_iou_thrs = len(self.params.iouThrs)  # T
        num_gt = len(gt)  # G
        num_dt = len(dt)  # D
        gtm = torch.zeros((num_iou_thrs, num_gt), device=self.device, requires_grad=self.requires_grad)
        dtm = torch.zeros((num_iou_thrs, num_dt), device=self.device, requires_grad=self.requires_grad)
        dt_ig = torch.zeros((num_iou_thrs, num_dt), device=self.device, requires_grad=self.requires_grad)
        if len(ious) != 0:
            for tind, t in enumerate(self.params.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = torch.min(
                        t,
                        torch.tensor(1, device=self.device, requires_grad=self.requires_grad)
                        - torch.finfo(torch.float32).eps,
                    )
                    m = -1
                    #############################################################################
                    # Vectorize the comparison and selection process
                    mask = (gtm[tind] <= 0) | iscrowd
                    valid_ious = ious[dind] * mask
                    m = int(torch.argmax(valid_ious).item())
                    #############################################################################

                    for gind, g in enumerate(gt):  # noqa: B007
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gt_ig[m] == 0 and gt_ig[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dt_ig[tind, dind] = gt_ig[m]
                    dtm[tind, dind] = gt[m].id
                    gtm[tind, m] = d.id
        # set unmatched detections outside of area range to ignore
        a = torch.Tensor([d.area < aRng[0] or d.area > aRng[1] for d in dt]).reshape((1, len(dt)))
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
            requires_grad=self.requires_grad,
        )  # -1 for the precision of absent categories
        recall = -torch.ones(
            (num_iou_thrs, num_cat_ids, num_area_rng, num_max_dets),
            device=self.device,
            requires_grad=self.requires_grad,
        )
        scores = -torch.ones(
            (num_iou_thrs, num_rec_thrs, num_cat_ids, num_area_rng, num_max_dets),
            device=self.device,
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
                    dt_scores = torch.concatenate([el.dtScores[0:max_det] for el in eval_img_results])

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

                    tp_sum = torch.cumsum(tps, dim=1).to(dtype=torch.float)
                    fp_sum = torch.cumsum(fps, dim=1).to(dtype=torch.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum, strict=False)):
                        # TODO: fix, why new tensors?
                        tp = tp.clone()
                        fp = fp.clone()
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + torch.finfo(torch.float32).eps)
                        q = torch.zeros((num_rec_thrs,), device=self.device, requires_grad=self.requires_grad)
                        ss = torch.zeros((num_rec_thrs,), device=self.device, requires_grad=self.requires_grad)

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = torch.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dt_scores_sorted[pi]
                        except:  # noqa: S110, E722 # nosec B110
                            pass  # TODO: fix this
                        precision[t, :, k, a, m] = torch.tensor(q, device=self.device, requires_grad=self.requires_grad)
                        scores[t, :, k, a, m] = ss.clone()
        self.eval = EvalResult(
            params=p,
            counts=[num_iou_thrs, num_rec_thrs, num_cat_ids, num_area_rng, num_max_dets],
            date=datetime.datetime.now(),  # .strftime("%Y-%m-%d %H:%M:%S"),
            precision=precision,
            recall=recall,
            scores=scores,
        )
        toc = time.time()
        self.logger.info(f"DONE (t={toc - tic:0.2f}s).")

    def summarize(self) -> None:
        """Compute and display summary metrics for evaluation results.

        Note:
            This functin can *only* be applied on the default parameter setting.
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
                f"{self.params.iouThrs[0]:0.2f}:{self.params.iouThrs[-1]:0.2f}" if iouThr is None else f"{iouThr:0.2f}"
            )

            aind = [i for i, aRng in enumerate(self.params.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(self.params.maxDets) if mDet == maxDets]
            iou_thr_tensor = torch.tensor(iouThr, device=self.device, requires_grad=self.requires_grad)
            if ap:
                # dimension of precision: [TxRxKxAxM]
                prec = self.eval.precision
                # IoU
                if iouThr is not None:
                    thr = torch.where(iou_thr_tensor == self.params.iouThrs)[0]
                    prec = prec[thr]
                prec = prec[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                prec = self.eval.recall
                if iouThr is not None:
                    thr = torch.where(iou_thr_tensor == self.params.iouThrs)[0]
                    prec = prec[thr]
                prec = prec[:, :, aind, mind]
            mean_s = (
                torch.tensor([-1.0], device=self.device, requires_grad=self.requires_grad)
                if len(prec[prec > -1]) == 0
                else torch.mean(prec[prec > -1])
            )
            self.logger.info(template.format(title, type_abbrv, iou, areaRng, maxDets, mean_s))
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
