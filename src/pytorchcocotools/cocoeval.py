from collections import defaultdict
import copy
import datetime
from logging import Logger
import time
from typing import Literal, TypeAlias

from pytorchcocotools import mask, utils
from pytorchcocotools.coco import COCO
import torch
from torch import Tensor

RangeLabel: TypeAlias = Literal["all", "small", "medium", "large"]
RangeLabels: TypeAlias = list[RangeLabel]
Ranges: TypeAlias = list[tuple[int, int]]
IoUType: TypeAlias = Literal["segm", "bbox", "keypoints"]


class Params:
    """Params for coco evaluation api."""

    def setDetParams(self):  # noqa: N802
        self.imgIds = []
        self.catIds = []
        # torch.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = torch.linspace(0.5, 0.95, int(round((0.95 - 0.5) / 0.05)) + 1)
        self.recThrs = torch.linspace(0.0, 1.00, int(round((1.00 - 0.0) / 0.01)) + 1)
        self.maxDets = [1, 10, 100]
        self.areaRng: Ranges = [
            (0**2, 1e5**2),
            (0**2, 32**2),
            (32**2, 96**2),
            (96**2, 1e5**2),
        ]
        self.areaRngLbl: RangeLabels = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setKpParams(self):  # noqa: N802
        self.imgIds = []
        self.catIds = []
        # torch.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = torch.linspace(0.5, 0.95, int(round((0.95 - 0.5) / 0.05)) + 1)
        self.recThrs = torch.linspace(0.0, 1.00, int(round((1.00 - 0.0) / 0.01)) + 1)
        self.maxDets = [20]
        self.areaRng: Ranges = [(0**2, 1e5**2), (32**2, 96**2), (96**2, 1e5**2)]
        self.areaRngLbl: RangeLabels = ["all", "medium", "large"]
        self.useCats = 1
        self.kpt_oks_sigmas = (
            torch.Tensor(
                [
                    0.26,
                    0.25,
                    0.25,
                    0.35,
                    0.35,
                    0.79,
                    0.79,
                    0.72,
                    0.72,
                    0.62,
                    0.62,
                    1.07,
                    1.07,
                    0.87,
                    0.87,
                    0.89,
                    0.89,
                ]
            )
            / 10.0
        )

    def __init__(self, iouType: IoUType = "segm"):  # noqa: N803
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        elif iouType == "keypoints":
            self.setKpParams()
        else:
            raise Exception("iouType not supported")  # noqa: TRY002
        self.iouType = iouType


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.

    logger: Logger

    def __init__(
        self,
        cocoGt: COCO = None,  # noqa: N803
        cocoDt: COCO = None,  # noqa: N803
        iouType: IoUType = "segm",  # noqa: N803
    ):
        """Initialize CocoEval using coco APIs for gt and dt.

        Args:
            cocoGt: COCO object with ground truth annotations. Defaults to None.
            cocoDt: COCO object with detection results. Defaults to None.
            iouType: _description_. Defaults to "segm".
        """
        self.logger = utils.get_logger(__name__)
        if not iouType:
            self.logger.info("iouType not specified. use default iouType segm")
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.eval_imgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params: Params = Params(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self) -> None:
        """Prepare ._gts and ._dts for evaluation based on params."""

        def _toMask(anns, coco: COCO):  # noqa: N802
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt.get("ignore", 0)
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.eval_imgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self) -> None:
        """Run per image evaluation on given images and store results (a list of dict) in self.evalImgs."""
        tic = time.time()
        self.logger.info("Running per image evaluation...")
        p = self.params
        self.logger.info(f"Evaluate annotation type *{p.iouType}*")
        p.imgIds = list(torch.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(torch.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        cat_ids = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            compute_iou = self.computeIoU
        elif p.iouType == "keypoints":
            compute_iou = self.computeOks
        self.ious = {(img_id, cat_id): compute_iou(img_id, cat_id) for img_id in p.imgIds for cat_id in cat_ids}

        evaluate_img = self.evaluateImg
        max_det = p.maxDets[-1]
        self.eval_imgs = [
            evaluate_img(imgId, catId, areaRng, max_det)
            for catId in cat_ids
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        self.logger.info(f"DONE (t={toc - tic:0.2f}s).")

    def computeIoU(self, imgId: int, catId: int) -> Tensor:  # noqa: N803, N802
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = torch.argsort(torch.Tensor([-d["score"] for d in dt]))
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("Unknown iouType for iou computation")  # noqa: TRY002

        # compute iou between each dt and gt region
        iscrowd = [bool(o["iscrowd"]) for o in gt]
        ious = mask.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId: int, catId: int) -> Tensor:  # noqa: N803, N802
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = torch.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0 : p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return torch.Tensor([])
        ious = torch.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = torch.Tensor(gt["keypoints"])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = torch.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = torch.Tensor(dt["keypoints"])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = torch.zeros(k)
                    dx = torch.max((z, x0 - xd), axis=0) + torch.max((z, xd - x1), axis=0)
                    dy = torch.max((z, y0 - yd), axis=0) + torch.max((z, yd - y1), axis=0)
                e = (dx**2 + dy**2) / vars / (gt["area"] + torch.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = torch.sum(torch.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId: int, catId: int, aRng: list, maxDet: int) -> dict:  # noqa: N803, N802
        """Perform evaluation for single category and image.

        Args:
            imgId: _description_
            catId: _description_
            aRng: _description_
            maxDet: _description_

        Returns:
            _description_
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for c_id in p.catIds for _ in self._gts[imgId, c_id]]
            dt = [_ for c_id in p.catIds for _ in self._dts[imgId, c_id]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = torch.argsort(Tensor([g["_ignore"] for g in gt]))
        gt = [gt[i] for i in gtind]
        dtind = torch.argsort(Tensor([-d["score"] for d in dt]))
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        num_iou_thrs = len(p.iouThrs)  # T
        num_gt = len(gt)  # G
        num_dt = len(dt)  # D
        gtm = torch.zeros((num_iou_thrs, num_gt))
        dtm = torch.zeros((num_iou_thrs, num_dt))
        gt_ig = torch.Tensor([g["_ignore"] for g in gt])
        dt_ig = torch.zeros((num_iou_thrs, num_dt))
        if len(ious) != 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - torch.eps])
                    m = -1
                    #############################################################################
                    # Vectorize the comparison and selection process
                    mask = (gtm[tind] <= 0) | iscrowd
                    valid_ious = ious[dind] * mask
                    m = torch.argmax(valid_ious).item()
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
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]
        # set unmatched detections outside of area range to ignore
        a = torch.Tensor([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dt_ig = torch.logical_or(dt_ig, torch.logical_and(dtm == 0, torch.repeat(a, num_iou_thrs, 0)))
        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gt_ig,
            "dtIgnore": dt_ig,
        }

    def accumulate(self, p: "Params" = None) -> None:
        """Accumulate per image evaluation results and store the result in self.eval.

        Args:
            p: Input params for evaluation. Defaults to None.
        """
        self.logger.info("Accumulating evaluation results...")
        tic = time.time()
        if not self.eval_imgs:
            self.logger.info("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        num_iou_thrs = len(p.iouThrs)  # T
        num_rec_thrs = len(p.recThrs)  # R
        num_cat_ids = len(p.catIds) if p.useCats else 1  # K
        num_area_rng = len(p.areaRng)  # A
        num_max_dets = len(p.maxDets)  # M
        precision = -torch.ones(
            (num_iou_thrs, num_rec_thrs, num_cat_ids, num_area_rng, num_max_dets)
        )  # -1 for the precision of absent categories
        recall = -torch.ones((num_iou_thrs, num_cat_ids, num_area_rng, num_max_dets))
        scores = -torch.ones((num_iou_thrs, num_rec_thrs, num_cat_ids, num_area_rng, num_max_dets))

        # create dictionary for future indexing
        _pe = self._paramsEval
        cat_ids = _pe.catIds if _pe.useCats else [-1]
        set_k = set(cat_ids)
        set_a = set(map(tuple, _pe.areaRng))
        set_m = set(_pe.maxDets)
        set_i = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in set_k]
        m_list = [m for n, m in enumerate(p.maxDets) if m in set_m]
        a_list = [n for n, a in enumerate(tuple(x) for x in p.areaRng) if a in set_a]
        i_list = [n for n, i in enumerate(p.imgIds) if i in set_i]
        i_0 = len(_pe.imgIds)
        a_0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            num_k = k0 * a_0 * i_0
            for a, a0 in enumerate(a_list):
                num_a = a0 * i_0
                for m, max_det in enumerate(m_list):
                    e = [self.eval_imgs[num_k + num_a + i] for i in i_list]
                    e = [e for e in e if e is not None]
                    if len(e) == 0:
                        continue
                    dt_scores = torch.concatenate([e["dtScores"][0:max_det] for e in e])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = torch.argsort(-dt_scores, kind="mergesort")
                    dt_scores_sorted = dt_scores[inds]

                    dt_matches = torch.concatenate([e["dtMatches"][:, 0:max_det] for e in e], axis=1)[:, inds]
                    dt_ig = torch.concatenate([e["dtIgnore"][:, 0:max_det] for e in e], axis=1)[:, inds]
                    gt_ig = torch.concatenate([e["gtIgnore"] for e in e])
                    npig = torch.count_nonzero(gt_ig == 0)
                    if npig == 0:
                        continue
                    tps = torch.logical_and(dt_matches, torch.logical_not(dt_ig))
                    fps = torch.logical_and(torch.logical_not(dt_matches), torch.logical_not(dt_ig))

                    tp_sum = torch.cumsum(tps, axis=1).to(dtype=torch.float)
                    fp_sum = torch.cumsum(fps, axis=1).to(dtype=torch.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum, strict=False)):
                        tp = torch.Tensor(tp)
                        fp = torch.Tensor(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + torch.spacing(1))
                        q = torch.zeros((num_rec_thrs,))
                        ss = torch.zeros((num_rec_thrs,))

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
                        except:  # noqa: S110, E722
                            pass
                        precision[t, :, k, a, m] = torch.Tensor(q)
                        scores[t, :, k, a, m] = torch.Tensor(ss)
        self.eval = {
            "params": p,
            "counts": [num_iou_thrs, num_rec_thrs, num_cat_ids, num_area_rng, num_max_dets],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        toc = time.time()
        self.logger.info(f"DONE (t={toc - tic:0.2f}s).")

    def summarize(self) -> None:
        """Compute and display summary metrics for evaluation results.

        Note:
            This functin can *only* be applied on the default parameter setting.
        """

        def _summarize(ap=1, iouThr: float = None, areaRng: RangeLabel = "all", maxDets: int = 100) -> int | Tensor:  # noqa: N803
            p = self.params
            template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            title = "Average Precision" if ap == 1 else "Average Recall"
            type_abbrv = "(AP)" if ap == 1 else "(AR)"
            iou = f"{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}" if iouThr is None else f"{iouThr:0.2f}"

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = torch.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = torch.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            mean_s = -1 if len(s[s > -1]) == 0 else torch.mean(s[s > -1])
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
        if iou_type == "segm" or iou_type == "bbox":
            summarize = _summarizeDets
        elif iou_type == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()
