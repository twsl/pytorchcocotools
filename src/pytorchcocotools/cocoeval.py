from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import copy
import datetime
from itertools import accumulate as running_sum
import logging
from logging import Logger
import time
from typing import cast

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools import mask
from pytorchcocotools.coco import COCO
from pytorchcocotools.internal.cocoeval_types import EvalImgResult, EvalResult, EvalState, Params
from pytorchcocotools.internal.entities import IoUObject, IoUType, RaggedRle, Range, RangeLabel, RleObj, TorchDevice
from pytorchcocotools.internal.mask_api._batch import (
    counts_to_strings,
    polygons_to_counts,
    ragged_take,
    rle_bboxes,
    run_bounds,
    segment_ids,
    segment_offsets,
    strings_to_counts,
    union_counts,
)
from pytorchcocotools.internal.structure.annotations import (
    CocoAnnotationDetection,
    CocoAnnotationKeypointDetection,
    CocoAnnotationObjectDetection,
    Segmentation,
)
from pytorchcocotools.utils.device import get_float_dtype
from pytorchcocotools.utils.logging import get_logger

_MAX_RLE_QUERIES = 1 << 24


class COCOeval:
    logger: Logger

    @staticmethod
    def _pair_entries(gt_pair: Tensor, pair_num_dt: Tensor, pair_dt_offset: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Enumerate every (detection rank, ground truth) combination inside the same pair.

        Args:
            gt_pair: The pair index of every ground truth.
            pair_num_dt: The number of detections per pair.
            pair_dt_offset: The flat index of the first detection of every pair.

        Returns:
            The detection rank, flat ground-truth index and flat detection index of every entry.
        """
        reps = pair_num_dt[gt_pair]
        entry_gt = segment_ids(reps)
        entry_d = torch.arange(entry_gt.numel(), device=gt_pair.device) - segment_offsets(reps)[entry_gt]
        entry_dt = pair_dt_offset[gt_pair[entry_gt]] + entry_d
        return entry_d, entry_gt, entry_dt

    @staticmethod
    def _bbox_iou(dt: Tensor, gt: Tensor, crowd: Tensor) -> Tensor:
        """Element-wise IoU of XYWH boxes, matching ``bbIou`` of pycocotools.

        Args:
            dt: The detection boxes, shape ``[E, 4]``.
            gt: The ground-truth boxes, shape ``[E, 4]``.
            crowd: Whether the ground truth is a crowd region, shape ``[E]``.

        Returns:
            The IoU per entry.
        """
        w = torch.minimum(dt[:, 2] + dt[:, 0], gt[:, 2] + gt[:, 0]) - torch.maximum(dt[:, 0], gt[:, 0])
        h = torch.minimum(dt[:, 3] + dt[:, 1], gt[:, 3] + gt[:, 1]) - torch.maximum(dt[:, 1], gt[:, 1])
        inter = w * h
        dt_area = dt[:, 2] * dt[:, 3]
        union = torch.where(crowd, dt_area, dt_area + gt[:, 2] * gt[:, 3] - inter)
        return torch.where((w > 0) & (h > 0), inter / union, torch.zeros_like(inter))

    @staticmethod
    def _oks(dt_kpts: Tensor, gt_kpts: Tensor, gt_box: Tensor, gt_area: Tensor, sigmas: Tensor) -> Tensor:
        """Element-wise object keypoint similarity, matching ``COCOeval.computeOks`` of pycocotools.

        Args:
            dt_kpts: The detection keypoints ``[x, y, v, ...]``, shape ``[E, 3K]``.
            gt_kpts: The ground-truth keypoints, shape ``[E, 3K]``.
            gt_box: The ground-truth XYWH boxes, shape ``[E, 4]``.
            gt_area: The ground-truth areas, shape ``[E]``.
            sigmas: The per-keypoint sigmas, shape ``[K]``.

        Returns:
            The OKS per entry.
        """
        xd, yd = dt_kpts[:, 0::3], dt_kpts[:, 1::3]
        xg, yg, vg = gt_kpts[:, 0::3], gt_kpts[:, 1::3], gt_kpts[:, 2::3]
        visible = vg > 0
        has_visible = visible.sum(1, keepdim=True) > 0
        x0 = (gt_box[:, 0] - gt_box[:, 2])[:, None]
        x1 = (gt_box[:, 0] + gt_box[:, 2] * 2)[:, None]
        y0 = (gt_box[:, 1] - gt_box[:, 3])[:, None]
        y1 = (gt_box[:, 1] + gt_box[:, 3] * 2)[:, None]
        dx = torch.where(has_visible, xd - xg, (x0 - xd).clamp(min=0) + (xd - x1).clamp(min=0))
        dy = torch.where(has_visible, yd - yg, (y0 - yd).clamp(min=0) + (yd - y1).clamp(min=0))
        variances = (sigmas * 2) ** 2
        e = (dx**2 + dy**2) / variances / (gt_area[:, None] + torch.finfo(gt_area.dtype).eps) / 2
        used = visible | ~has_visible
        return torch.where(used, torch.exp(-e), 0).sum(1) / used.sum(1)

    @staticmethod
    def _rle_iou(dt: RaggedRle, gt: RaggedRle, entry_dt: Tensor, entry_gt: Tensor, crowd: Tensor) -> Tensor:
        """Element-wise IoU of RLE masks, matching ``rleIou`` of pycocotools.

        The intersection is computed without decoding: for every foreground run ``[s, e)`` of the ground
        truth, ``cov(e) - cov(s)`` is added, where ``cov(x)`` counts detection foreground pixels before
        ``x``. ``cov`` is evaluated with one ``searchsorted`` over all detection runs.

        Args:
            dt: The detection masks.
            gt: The ground-truth masks.
            entry_dt: The detection mask of every entry.
            entry_gt: The ground-truth mask of every entry.
            crowd: Whether the ground truth is a crowd region, per entry.

        Returns:
            The IoU per entry, in the float dtype of the masks' device.
        """
        device = dt.counts.device
        num_entries = entry_dt.numel()
        dt_starts, dt_fg, dt_owner = run_bounds(dt.counts, dt.lengths)
        dt_fg_pixels = torch.where(dt_fg, dt.counts, 0)
        csum = torch.cumsum(dt_fg_pixels, 0) - dt_fg_pixels
        dt_cov = csum - csum[segment_offsets(dt.lengths).clamp(max=max(csum.numel() - 1, 0))][dt_owner]
        dt_area = dt.counts.new_zeros(dt.lengths.numel()).index_add_(0, dt_owner, dt_fg_pixels)

        gt_starts, gt_fg, gt_owner = run_bounds(gt.counts, gt.lengths)
        gt_fg &= gt.counts > 0
        gt_run_s = gt_starts[gt_fg]
        gt_run_e = gt_run_s + gt.counts[gt_fg]
        gt_num_runs = torch.bincount(gt_owner[gt_fg], minlength=gt.lengths.numel())
        gt_area = gt.counts.new_zeros(gt.lengths.numel()).index_add_(0, gt_owner[gt_fg], gt.counts[gt_fg])

        db = rle_bboxes(dt.counts, dt.lengths, dt.h)[entry_dt]
        gb = rle_bboxes(gt.counts, gt.lengths, gt.h)[entry_gt]
        overlap = (torch.minimum(db[:, 2], gb[:, 2]) >= torch.maximum(db[:, 0], gb[:, 0])) & (
            torch.minimum(db[:, 3], gb[:, 3]) >= torch.maximum(db[:, 1], gb[:, 1])
        )
        same_size = (dt.h[entry_dt] == gt.h[entry_gt]) & (dt.w[entry_dt] == gt.w[entry_gt])
        todo = (overlap & same_size).nonzero().squeeze(1)

        big = int((dt.h * dt.w).max()) + 2 if dt.h.numel() else 2
        run_keys = dt_owner * big + dt_starts
        inter = dt.counts.new_zeros(num_entries)
        runs_per_todo = gt_num_runs[entry_gt[todo]]
        query_end = torch.cumsum(runs_per_todo, 0)
        start = 0
        while start < todo.numel():
            budget = (query_end[start] - runs_per_todo[start] + _MAX_RLE_QUERIES).view(1)
            stop = max(int(torch.searchsorted(query_end, budget, right=True)), start + 1)
            chunk = todo[start:stop]
            reps = runs_per_todo[start:stop]
            q_entry = torch.repeat_interleave(chunk, reps)
            q_local = torch.arange(q_entry.numel(), device=device) - segment_offsets(reps).repeat_interleave(reps)
            q_run = segment_offsets(gt_num_runs)[entry_gt[q_entry]] + q_local
            owner_key = entry_dt[q_entry] * big
            covered = torch.zeros_like(q_entry)
            for q, sign in ((gt_run_e[q_run], 1), (gt_run_s[q_run], -1)):
                k = torch.searchsorted(run_keys, owner_key + q, right=True) - 1
                partial = torch.minimum(q - dt_starts[k], dt.counts[k])
                covered += sign * (dt_cov[k] + torch.where(dt_fg[k], partial, 0))
            inter.index_add_(0, q_entry, covered)
            start = stop

        fdtype = get_float_dtype(device)
        inter_f = inter.to(fdtype)
        area_d = dt_area[entry_dt].to(fdtype)
        union = torch.where(crowd, area_d, area_d + gt_area[entry_gt].to(fdtype) - inter_f)
        iou = torch.where(union > 0, inter_f / union.clamp(min=1), torch.zeros_like(inter_f))
        return torch.where(overlap & ~same_size, -1.0, iou)

    @staticmethod
    def _greedy_match(
        iou: Tensor,
        gt_pair: Tensor,
        gt_ignore: Tensor,
        gt_crowd: Tensor,
        pair_num_dt: list[int],
        pair_gt_end: list[int],
        pair_dt_offset: Tensor,
        num_dt: int,
        thresholds: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Greedy COCO matching for all pairs, area ranges and IoU thresholds at once.

        Args:
            iou: IoU of detection rank ``d`` with every ground truth, shape ``[D, N]`` (``-1`` if absent).
            gt_pair: The pair index of every ground truth (non-decreasing).
            gt_ignore: The ignore flag per area range and ground truth, shape ``[A, N]``.
            gt_crowd: The crowd flag per ground truth.
            pair_num_dt: The number of detections per pair (non-increasing).
            pair_gt_end: The flat end offset of the ground truths of every pair.
            pair_dt_offset: The flat index of the first detection of every pair.
            num_dt: The total number of detections.
            thresholds: The IoU thresholds, shape ``[T]``.

        Returns:
            ``dt_gt[A, T, num_dt]``: the matched flat ground-truth index per detection (``-1`` if none) and
            ``gt_dt[A, T, N]``: the last matched flat detection index per ground truth (``-1`` if none).
        """
        device = iou.device
        num_a, num_gt = gt_ignore.shape
        num_t = thresholds.numel()
        dt_gt = torch.full((num_a, num_t, num_dt), -1, dtype=torch.long, device=device)
        # one extra column absorbs the scatters of unmatched pairs
        gt_dt = torch.full((num_a, num_t, num_gt + 1), -1, dtype=torch.long, device=device)
        matched = torch.zeros((num_a, num_t, num_gt + 1), dtype=torch.bool, device=device)
        non_ignored = ~gt_ignore[:, None, :]
        thr = thresholds[None, :, None]
        position = torch.arange(num_gt, device=device)
        crowd_or_trash = torch.cat([gt_crowd, gt_crowd.new_ones(1)])

        active_pairs = len(pair_num_dt)
        for d in range(max(pair_num_dt, default=0)):
            while active_pairs > 0 and pair_num_dt[active_pairs - 1] <= d:
                active_pairs -= 1
            p, n = active_pairs, pair_gt_end[active_pairs - 1]
            if n == 0:
                continue
            index = gt_pair[:n].expand(num_a, num_t, n)
            cand = (iou[d, :n] >= thr) & ~matched[:, :, :n]
            ni = non_ignored[:, :, :n]
            has_ni = torch.zeros((num_a, num_t, p), dtype=torch.int8, device=device)
            has_ni.scatter_reduce_(2, index, (cand & ni).to(torch.int8), reduce="amax")
            sel = cand & (ni == (has_ni.gather(2, index) > 0))
            score = torch.where(sel, iou[d, :n], -torch.inf)
            best = torch.full((num_a, num_t, p), -torch.inf, dtype=iou.dtype, device=device)
            best.scatter_reduce_(2, index, score, reduce="amax")
            # ties go to the last ground truth, as in the reference loop
            last = torch.where(sel & (score == best.gather(2, index)), position[:n], -1)
            choice = torch.full((num_a, num_t, p), -1, dtype=torch.long, device=device)
            choice.scatter_reduce_(2, index, last, reduce="amax")

            dt_index = pair_dt_offset[:p] + d
            dt_gt[:, :, dt_index] = choice
            target = torch.where(choice >= 0, choice, num_gt)
            gt_dt.scatter_(2, target, dt_index.expand(num_a, num_t, p))
            matched.scatter_(2, torch.where(crowd_or_trash[target], num_gt, target), True)
        return dt_gt, gt_dt[:, :, :num_gt]

    @staticmethod
    def _first_reaching(rec_thrs: Tensor, npig: Tensor) -> Tensor:
        """Smallest integer ``t`` with ``t / npig >= r`` in floating point, per ``npig`` and threshold ``r``."""
        npig_f = npig.to(rec_thrs.dtype)[..., None]
        t0 = torch.ceil(rec_thrs * npig_f)
        tau = torch.where((t0 - 1) / npig_f >= rec_thrs, t0 - 1, torch.where(t0 / npig_f >= rec_thrs, t0, t0 + 1))
        return tau.clamp(min=0).long()

    @staticmethod
    def _accumulate(
        dt_cat: Tensor,
        dt_order_key: Tensor,
        dt_rank: Tensor,
        dt_score: Tensor,
        dt_tp: Tensor,
        dt_ignore: Tensor,
        npig: Tensor,
        max_dets: list[int],
        rec_thrs: Tensor,
        num_cats: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute precision, recall and score tables for all categories, area ranges and max detections.

        Args:
            dt_cat: The output category of every detection (``-1`` to exclude it).
            dt_order_key: Tie-break key reproducing the reference concatenation order (image, rank).
            dt_rank: The rank of every detection inside its pair.
            dt_score: The detection scores.
            dt_tp: Whether the detection matched a ground truth, shape ``[A, T, N]``.
            dt_ignore: Whether the detection is ignored, shape ``[A, T, N]``.
            npig: The number of non-ignored ground truths, shape ``[A, K]``.
            max_dets: The max-detection limits.
            rec_thrs: The recall thresholds, shape ``[R]``.
            num_cats: The number of output categories ``K``.

        Returns:
            ``precision[T, R, K, A, M]``, ``recall[T, K, A, M]`` and ``scores[T, R, K, A, M]``.
        """
        device = dt_score.device
        dtype = rec_thrs.dtype
        num_a, num_t = dt_tp.shape[:2]
        num_r, num_m = rec_thrs.numel(), len(max_dets)
        precision = torch.full((num_t, num_r, num_cats, num_a, num_m), -1.0, dtype=dtype, device=device)
        recall = torch.full((num_t, num_cats, num_a, num_m), -1.0, dtype=dtype, device=device)
        scores = torch.full_like(precision, -1.0)

        order = torch.argsort(dt_order_key, stable=True)
        order = order[torch.argsort(dt_score[order], descending=True, stable=True)]
        order = order[torch.argsort(dt_cat[order], stable=True)]
        order = order[dt_cat[order] >= 0]
        has_gt = npig > 0  # [A, K]
        tau = COCOeval._first_reaching(rec_thrs, npig)  # [A, K, R]
        eps = torch.finfo(dtype).eps
        cat_range = torch.arange(num_cats, device=device)

        for m, max_det in enumerate(max_dets):
            sel = order[dt_rank[order] < max_det]
            n = sel.numel()
            cat = dt_cat[sel]
            nd = torch.bincount(cat, minlength=num_cats)
            seg_end = torch.cumsum(nd, 0)
            seg_start = seg_end - nd
            if n == 0:
                fill = torch.where(has_gt, 0.0, -1.0).T
                recall[:, :, :, m] = fill[None]
                precision[:, :, :, :, m] = fill[None, None]
                scores[:, :, :, :, m] = fill[None, None]
                continue

            tp_ok, ig = dt_tp[:, :, sel], dt_ignore[:, :, sel]
            tps = tp_ok & ~ig
            fps = ~tp_ok & ~ig
            tp = torch.cumsum(tps, -1, dtype=dtype)
            fp = torch.cumsum(fps, -1, dtype=dtype)
            start_idx = seg_start.clamp(max=n - 1)
            tp = tp - (tp - tps.to(dtype))[:, :, start_idx][:, :, cat]
            fp = fp - (fp - fps.to(dtype))[:, :, start_idx][:, :, cat]

            rc = tp / npig[:, None, cat].to(dtype)
            pr = tp / (fp + tp + eps)

            last = torch.where(nd > 0, seg_end - 1, 0)
            rc_last = torch.where(nd > 0, rc[:, :, last], 0.0)  # [A, T, K]
            recall[:, :, :, m] = torch.where(has_gt[:, None, :], rc_last, -1.0).permute(1, 2, 0)

            # suffix maximum of precision inside every category segment
            rank = torch.argsort(torch.argsort(pr, -1), -1)
            key = (num_cats - 1 - cat) * n + rank
            _, idx = torch.cummax(key.flip(-1), -1)
            pr_mono = pr.gather(-1, (n - 1 - idx).flip(-1))

            # recall thresholds reached: count of segment entries with tp < tau
            big = n + 2
            seq = cat * big + tp.long()
            query = (cat_range[:, None] * big + tau.clamp(max=n + 1))[:, None].expand(num_a, num_t, num_cats, num_r)
            pos = torch.searchsorted(seq, query.reshape(num_a, num_t, -1)).view(num_a, num_t, num_cats, num_r)
            valid = pos < seg_end[:, None]
            pos_c = pos.clamp(max=n - 1)
            q = torch.where(valid, pr_mono.gather(-1, pos_c.view(num_a, num_t, -1)).view_as(pos), 0.0)
            ss = torch.where(valid, dt_score[sel][pos_c], 0.0).to(dtype)
            mask = has_gt[:, None, :, None]
            precision[:, :, :, :, m] = torch.where(mask, q, -1.0).permute(1, 3, 2, 0)
            scores[:, :, :, :, m] = torch.where(mask, ss, -1.0).permute(1, 3, 2, 0)
        return precision, recall, scores

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
        self._state: EvalState | None = None
        self._gt_rle_row: dict[int, int] = {}
        self._dt_rle_row: dict[int, int] = {}
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
        self.float_dtype = get_float_dtype(device)
        self.requires_grad = requires_grad if requires_grad is not None else False
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    @property
    def eval_imgs(self) -> list[EvalImgResult | None]:
        """Per-image per-category evaluation results ``[K x A x I]``, materialised on first access."""
        if self._eval_imgs is None:
            self._eval_imgs = self._build_eval_imgs() if self._state is not None else []
        return self._eval_imgs

    @eval_imgs.setter
    def eval_imgs(self, value: list[EvalImgResult | None] | None) -> None:
        self._eval_imgs = value

    @property
    def ious(self) -> dict[tuple[int, int], Tensor]:
        """IoUs between detections and ground truths per ``(imgId, catId)``, materialised on first access."""
        if self._ious is None:
            self._ious = self._build_ious() if self._state is not None else {}
        return self._ious

    @ious.setter
    def ious(self, value: dict[tuple[int, int], Tensor] | None) -> None:
        self._ious = value

    def _segm_to_rle(self, anns: list[CocoAnnotationDetection], coco: COCO) -> RaggedRle:
        """Convert the segmentations of ``anns`` to RLE in one batch and store them on the annotations.

        Mirrors ``ann['segmentation'] = coco.annToRLE(ann)`` of the reference implementation while keeping
        the decoded counts for the IoU kernel, so masks never round-trip through strings.
        """
        num = len(anns)
        hs, ws = [0] * num, [0] * num
        poly_xy: list[float] = []
        poly_vertices: list[int] = []
        poly_owner: list[int] = []
        raw_counts: list[int] = []
        raw_lengths: list[int] = []
        raw_owner: list[int] = []
        strings: list[bytes] = []
        string_owner: list[int] = []
        for i, ann in enumerate(anns):
            segm = ann.segmentation
            if isinstance(segm, list) and segm and isinstance(segm[0], list) and len(segm[0]) > 4:
                img = coco.imgs[ann.image_id]
                hs[i], ws[i] = img.height, img.width
                for poly in segm:
                    num_vertices = len(poly) // 2
                    poly_xy.extend(poly[: 2 * num_vertices])
                    poly_vertices.append(num_vertices)
                    poly_owner.append(i)
            elif isinstance(segm, dict) and isinstance(segm["counts"], list):
                hs[i], ws[i] = segm["size"]
                raw_counts.extend(segm["counts"])
                raw_lengths.append(len(segm["counts"]))
                raw_owner.append(i)
            else:
                if not isinstance(segm, dict):
                    # boxes and other rare inputs keep the scalar reference path
                    rle = coco.annToRLE(ann)
                    segm = rle[0] if isinstance(rle, list) else rle
                    ann.segmentation = cast(list[Segmentation], segm)  # ty:ignore[invalid-assignment]
                hs[i], ws[i] = segm["size"]
                counts = segm["counts"]
                strings.append(counts.encode() if isinstance(counts, str) else counts)
                string_owner.append(i)

        dev = self.device
        h = torch.tensor(hs, dtype=torch.long, device=dev)
        w = torch.tensor(ws, dtype=torch.long, device=dev)
        parts: list[tuple[Tensor, Tensor, Tensor]] = []
        if poly_vertices:
            owner = torch.tensor(poly_owner, device=dev)
            counts, lengths = polygons_to_counts(
                torch.tensor(poly_xy, dtype=self.float_dtype, device=dev).view(-1, 2),
                torch.tensor(poly_vertices, device=dev),
                h[owner],
                w[owner],
            )
            owners, group, polys_per_owner = torch.unique(owner, return_inverse=True, return_counts=True)
            single = (polys_per_owner[group] == 1).nonzero().squeeze(1)
            parts.append((*ragged_take(counts, lengths, single), owner[single]))
            multi = (polys_per_owner[group] > 1).nonzero().squeeze(1)
            if multi.numel():
                multi_owners, multi_group = torch.unique(owner[multi], return_inverse=True)
                merged = union_counts(*ragged_take(counts, lengths, multi), multi_group, (h * w)[multi_owners])
                parts.append((*merged, multi_owners))
        if raw_lengths:
            parts.append(
                (
                    torch.tensor(raw_counts, dtype=torch.long, device=dev),
                    torch.tensor(raw_lengths, device=dev),
                    torch.tensor(raw_owner, device=dev),
                )
            )
        if strings:
            counts, lengths = strings_to_counts(strings, device=dev)
            parts.append((counts, lengths, torch.tensor(string_owner, device=dev)))
        if not parts:
            empty = torch.zeros(0, dtype=torch.long, device=dev)
            return RaggedRle(empty, empty, h, w)

        all_counts = torch.cat([c for c, _, _ in parts])
        all_lengths = torch.cat([n for _, n, _ in parts])
        counts, lengths = ragged_take(all_counts, all_lengths, torch.argsort(torch.cat([o for _, _, o in parts])))

        converted = sorted(set(poly_owner)) + raw_owner
        if converted:
            rows = torch.tensor(converted, device=dev)
            for i, s in zip(converted, counts_to_strings(*ragged_take(counts, lengths, rows)), strict=True):
                anns[i].segmentation = cast(list[Segmentation], RleObj(size=[hs[i], ws[i]], counts=s))  # ty:ignore[invalid-assignment]
        return RaggedRle(counts, lengths, h, w)

    def _prepare(self) -> None:
        """Prepare ._gts and ._dts for evaluation based on params."""
        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            self._gt_rle = self._segm_to_rle(gts, self.cocoGt)
            self._dt_rle = self._segm_to_rle(dts, self.cocoDt)
            self._gt_rle_row = {id(g): i for i, g in enumerate(gts)}
            self._dt_rle_row = {id(d): i for i, d in enumerate(dts)}
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

        # Pre-sort DTs by score descending per (imgId, catId) and limit to maxDets
        max_det = self.params.maxDets[-1] if self.params.maxDets else 100
        for key in self._dts:
            dt_list = self._dts[key]
            if len(dt_list) > 1:
                dt_list.sort(key=lambda d: -(d.score if d.score is not None else 0))
            if len(dt_list) > max_det:
                self._dts[key] = dt_list[:max_det]

        self._state = None
        self.eval_imgs = []
        self.ious = {}
        self.eval = EvalResult()

    def evaluate(self) -> None:
        """Run per image evaluation on given images and store results (a list of dict) in self.evalImgs."""
        tic = time.time()
        self.logger.info("Running per image evaluation...")
        p = self.params
        p.iouThrs = p.iouThrs.to(dtype=self.float_dtype)
        p.recThrs = p.recThrs.to(dtype=self.float_dtype)
        if p.iouType == "keypoints":
            p.kpt_oks_sigmas = p.kpt_oks_sigmas.to(dtype=self.float_dtype)
        self.logger.info(f"Evaluate annotation type *{p.iouType}*")
        p.imgIds = sorted(set(p.imgIds))
        if p.useCats:
            p.catIds = sorted(set(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        self._state = self._evaluate_all()
        self.eval_imgs = None
        self.ious = None

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        self.logger.info(f"DONE (t={toc - tic:0.2f}s).")

    def _collect_pairs(
        self, cat_ids: list[int], max_det: int
    ) -> list[tuple[int, int, list[CocoAnnotationDetection], list[CocoAnnotationDetection]]]:
        """Return ``(cat index, img index, gts, dts)`` for every non-empty pair in reference order."""
        p = self.params
        img_index = {img_id: i for i, img_id in enumerate(p.imgIds)}
        if p.useCats:
            cat_index = {cat_id: k for k, cat_id in enumerate(cat_ids)}
            keys = {key for key, v in self._gts.items() if v} | {key for key, v in self._dts.items() if v}
            keys = sorted((cat_index[c], img_index[i]) for i, c in keys if c in cat_index and i in img_index)
            return [
                (k, i, self._gts.get((p.imgIds[i], cat_ids[k]), []), self._dts.get((p.imgIds[i], cat_ids[k]), []))
                for k, i in keys
            ]
        pairs = []
        for i, img_id in enumerate(p.imgIds):
            gts = [g for c in p.catIds for g in self._gts.get((img_id, c), [])]
            dts = [d for c in p.catIds for d in self._dts.get((img_id, c), [])]
            dts = sorted(dts, key=lambda d: -(d.score if d.score is not None else 0))[:max_det]
            if gts or dts:
                pairs.append((0, i, gts, dts))
        return pairs

    def _pair_ious(
        self,
        gts: list[CocoAnnotationDetection],
        dts: list[CocoAnnotationDetection],
        entry_gt: Tensor,
        entry_dt: Tensor,
        crowd: Tensor,
    ) -> Tensor:
        """IoU / OKS of every (detection, ground truth) entry, on ``self.device`` in ``self.float_dtype``."""
        p = self.params
        dev, fdtype = self.device, self.float_dtype
        if p.iouType == "bbox":
            gt_box = torch.tensor([g.bbox for g in gts], dtype=fdtype, device=dev).view(-1, 4)
            dt_box = torch.tensor([d.bbox for d in dts], dtype=fdtype, device=dev).view(-1, 4)
            return self._bbox_iou(dt_box[entry_dt], gt_box[entry_gt], crowd)
        if p.iouType == "segm":
            gt_rows = torch.tensor([self._gt_rle_row[id(g)] for g in gts], dtype=torch.long, device=dev)
            dt_rows = torch.tensor([self._dt_rle_row[id(d)] for d in dts], dtype=torch.long, device=dev)
            return self._rle_iou(self._dt_rle, self._gt_rle, dt_rows[entry_dt], gt_rows[entry_gt], crowd)
        if p.iouType == "keypoints" and p.useCats:
            kp_gts = cast(list[CocoAnnotationKeypointDetection], gts)
            kp_dts = cast(list[CocoAnnotationKeypointDetection], dts)
            gt_kpts = torch.tensor([g.keypoints for g in kp_gts], dtype=fdtype, device=dev)
            dt_kpts = torch.tensor([d.keypoints for d in kp_dts], dtype=fdtype, device=dev)
            gt_box = torch.tensor([g.bbox for g in kp_gts], dtype=fdtype, device=dev).view(-1, 4)
            gt_area = torch.tensor([g.area for g in kp_gts], dtype=fdtype, device=dev)
            sigmas = p.kpt_oks_sigmas.to(dtype=fdtype, device=dev)
            return self._oks(dt_kpts[entry_dt], gt_kpts[entry_gt], gt_box[entry_gt], gt_area[entry_gt], sigmas)
        # the reference computes no keypoint similarities when categories are merged
        return torch.full((entry_gt.numel(),), -1.0, dtype=fdtype, device=dev)

    def _evaluate_all(self) -> EvalState:
        """Compute IoUs and greedy matches for all (image, category) pairs in one batch."""
        p = self.params
        dev, fdtype = self.device, self.float_dtype
        cat_ids = p.catIds if p.useCats else [-1]
        max_det = p.maxDets[-1]
        pairs = self._collect_pairs(cat_ids, max_det)
        pairs.sort(key=lambda pair: -len(pair[3]))  # stable: keeps (cat, img) order within equal counts

        gts = [g for _, _, gl, _ in pairs for g in gl]
        dts = [d for _, _, _, dl in pairs for d in dl]
        pair_num_gt = [len(gl) for _, _, gl, _ in pairs]
        pair_num_dt = [len(dl) for _, _, _, dl in pairs]
        pair_gt_end = list(running_sum(pair_num_gt))
        pair_dt_end = list(running_sum(pair_num_dt))
        pair_gt_offset = [e - n for e, n in zip(pair_gt_end, pair_num_gt, strict=True)]
        pair_dt_offset = [e - n for e, n in zip(pair_dt_end, pair_num_dt, strict=True)]
        num_pairs_t = torch.arange(len(pairs), device=dev)
        pair_num_dt_t = torch.tensor(pair_num_dt, dtype=torch.long, device=dev)
        gt_pair = torch.repeat_interleave(num_pairs_t, torch.tensor(pair_num_gt, dtype=torch.long, device=dev))
        dt_pair = torch.repeat_interleave(num_pairs_t, pair_num_dt_t)
        dt_offset_t = torch.tensor(pair_dt_offset, dtype=torch.long, device=dev)

        gt_crowd = torch.tensor([bool(g.iscrowd) for g in gts], dtype=torch.bool, device=dev)
        entry_d, entry_gt, entry_dt = self._pair_entries(gt_pair, pair_num_dt_t, dt_offset_t)
        iou = torch.full((max(pair_num_dt, default=0), len(gts)), -1.0, dtype=fdtype, device=dev)
        iou[entry_d, entry_gt] = self._pair_ious(gts, dts, entry_gt, entry_dt, gt_crowd[entry_gt])

        gt_ids = torch.tensor([g.id for g in gts], dtype=torch.long, device=dev)
        gt_area = torch.tensor([g.area for g in gts], dtype=fdtype, device=dev)
        gt_ign = torch.tensor([bool(g.ignore) for g in gts], dtype=torch.bool, device=dev)
        dt_ids = torch.tensor([d.id for d in dts], dtype=torch.long, device=dev)
        dt_scores = torch.tensor(
            [float(d.score) if d.score is not None else 0.0 for d in dts], dtype=fdtype, device=dev
        )
        dt_area = torch.tensor([d.area for d in dts], dtype=fdtype, device=dev)
        a_rng = torch.tensor(p.areaRng, dtype=fdtype, device=dev).view(-1, 2)
        gt_ignore = gt_ign[None] | (gt_area[None] < a_rng[:, :1]) | (gt_area[None] > a_rng[:, 1:])
        thresholds = torch.tensor([min(t, 1 - 1e-10) for t in p.iouThrs.tolist()], dtype=fdtype, device=dev)

        dt_gt, gt_dt = self._greedy_match(
            iou, gt_pair, gt_ignore, gt_crowd, pair_num_dt, pair_gt_end, dt_offset_t, len(dts), thresholds
        )
        num_a, num_t = gt_ignore.shape[0], thresholds.numel()
        matched = dt_gt >= 0
        safe = dt_gt.clamp(min=0)
        # a match to ground-truth id 0 counts as unmatched in the reference (``dtm == 0``)
        dt_tp = matched & (gt_ids[safe] != 0) if len(gts) else matched
        gt_ig_at = gt_ignore.gather(1, safe.view(num_a, -1)).view_as(safe) if len(gts) else matched
        dt_area_out = (dt_area[None] < a_rng[:, :1]) | (dt_area[None] > a_rng[:, 1:])
        dt_ignore = (matched & gt_ig_at) | (~dt_tp & dt_area_out[:, None, :].expand(num_a, num_t, -1))

        return EvalState(
            cat_ids=list(cat_ids),
            img_ids=list(p.imgIds),
            area_rngs=list(p.areaRng),
            max_det=max_det,
            pair_cat=[k for k, _, _, _ in pairs],
            pair_img=[i for _, i, _, _ in pairs],
            pair_num_dt=pair_num_dt,
            pair_num_gt=pair_num_gt,
            pair_dt_offset=pair_dt_offset,
            pair_gt_offset=pair_gt_offset,
            dt_pair=dt_pair,
            dt_rank=torch.arange(len(dts), device=dev) - dt_offset_t[dt_pair],
            dt_ids=dt_ids,
            dt_scores=dt_scores,
            gt_pair=gt_pair,
            gt_ids=gt_ids,
            gt_ignore=gt_ignore,
            iou=iou,
            dt_gt=dt_gt,
            gt_dt=gt_dt,
            dt_tp=dt_tp,
            dt_ignore=dt_ignore,
        )

    def _build_ious(self) -> dict[tuple[int, int], Tensor]:
        s = cast(EvalState, self._state)
        empty = torch.empty(0, dtype=self.float_dtype, device=self.device)
        ious = {(img_id, cat_id): empty for img_id in s.img_ids for cat_id in s.cat_ids}
        for pi, (k, i) in enumerate(zip(s.pair_cat, s.pair_img, strict=True)):
            num_dt, num_gt, g0 = s.pair_num_dt[pi], s.pair_num_gt[pi], s.pair_gt_offset[pi]
            if num_dt and num_gt and (self.params.iouType != "keypoints" or self.params.useCats):
                ious[s.img_ids[i], s.cat_ids[k]] = s.iou[:num_dt, g0 : g0 + num_gt]
        return ious

    def _build_eval_imgs(self) -> list[EvalImgResult | None]:
        s = cast(EvalState, self._state)
        dt_match_ids = torch.where(s.dt_gt >= 0, s.gt_ids[s.dt_gt.clamp(min=0)], 0) if s.gt_ids.numel() else s.dt_gt
        gt_match_ids = torch.where(s.gt_dt >= 0, s.dt_ids[s.gt_dt.clamp(min=0)], 0) if s.dt_ids.numel() else s.gt_dt
        # ground truths per pair sorted with ignored ones last, as the reference does per area range
        gt_order = torch.argsort(s.gt_pair[None] * 2 + s.gt_ignore.long(), dim=1, stable=True)
        lookup = {(k, i): pi for pi, (k, i) in enumerate(zip(s.pair_cat, s.pair_img, strict=True))}
        results: list[EvalImgResult | None] = []
        for k, cat_id in enumerate(s.cat_ids):
            for a, a_rng in enumerate(s.area_rngs):
                for i, img_id in enumerate(s.img_ids):
                    pi = lookup.get((k, i))
                    if pi is None:
                        results.append(None)
                        continue
                    d0, d1 = s.pair_dt_offset[pi], s.pair_dt_offset[pi] + s.pair_num_dt[pi]
                    order = gt_order[a, s.pair_gt_offset[pi] : s.pair_gt_offset[pi] + s.pair_num_gt[pi]]
                    results.append(
                        EvalImgResult(
                            image_id=img_id,
                            category_id=cat_id,
                            aRng=a_rng,
                            maxDet=s.max_det,
                            dtIds=s.dt_ids[d0:d1],
                            gtIds=s.gt_ids[order],
                            dtMatches=dt_match_ids[a, :, d0:d1],
                            gtMatches=gt_match_ids[a][:, order],
                            dtScores=s.dt_scores[d0:d1],
                            gtIgnore=s.gt_ignore[a, order],
                            dtIgnore=s.dt_ignore[a, :, d0:d1],
                        )
                    )
        return results

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

        if len(gt) == 0 or len(dt) == 0:
            return torch.empty(0, dtype=self.float_dtype, device=self.device)
        if not self.params.useCats:
            # Combined list needs sorting; per-category lists are pre-sorted in _prepare()
            inds = torch.argsort(
                torch.tensor(
                    [-d.score if d.score is not None else 0 for d in dt],
                    device=self.device,
                    requires_grad=self.requires_grad,
                )
            )
            dt = [dt[i] for i in inds.tolist()]
            if len(dt) > self.params.maxDets[-1]:
                dt = dt[0 : self.params.maxDets[-1]]
        # DTs already pre-sorted and limited in _prepare() for useCats=True

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
        #     poly = tv.KeyPoints(
        #         torch.tensor(
        #             segmentation, dtype=torch.float64, device=self.device, requires_grad=self.requires_grad
        #         ).view(-1, 2),
        #         canvas_size=(size[0], size[1]),
        #         device=self.device,
        #         requires_grad=self.requires_grad,
        #     )
        #     return rleFrPoly(poly, device=self.device, requires_grad=self.requires_grad)

        img = self.cocoGt.loadImgs(imgId)[0]
        size = img.height, img.width

        if self.params.iouType == "segm":
            g = cast(IoUObject, [to_internal_types(g.segmentation, size) for g in gt])
            d = cast(IoUObject, [to_internal_types(d.segmentation, size) for d in dt])
        elif self.params.iouType == "bbox":
            g = tv.BoundingBoxes(
                torch.tensor(
                    [g.bbox for g in gt], dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad
                ),
                format=tv.BoundingBoxFormat.XYWH,
                canvas_size=size,
                device=self.device,
                requires_grad=self.requires_grad,
            )  # ty:ignore[no-matching-overload]
            d = tv.BoundingBoxes(
                torch.tensor(
                    [d.bbox for d in dt], dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad
                ),
                format=tv.BoundingBoxFormat.XYWH,
                canvas_size=size,
                device=self.device,
                requires_grad=self.requires_grad,
            )  # ty:ignore[no-matching-overload]
            if g.numel() == 0 or d.numel() == 0:
                return torch.empty(0, dtype=self.float_dtype, device=self.device)
        else:
            raise ValueError("Unknown iouType for iou computation.")  # noqa: TRY002

        # compute iou between each dt and gt region
        iscrowd = [bool(o.iscrowd) for o in gt]
        ious = mask.iou(d, g, iscrowd, device=self.device, requires_grad=self.requires_grad)
        return ious.to(dtype=self.float_dtype)

    @torch.inference_mode()
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
            return torch.empty(0, dtype=self.float_dtype, device=self.device)
        ious = torch.zeros(
            (len(dts), len(gts)), dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad
        )
        sigmas = self.params.kpt_oks_sigmas.to(device=self.device, dtype=self.float_dtype)
        vars = (sigmas * 2) ** 2
        k = len(sigmas)

        # Optimized: Vectorize OKS computation
        # Pre-convert all keypoints and bboxes to tensors
        if len(gts) > 0 and len(dts) > 0:
            # Stack all gt keypoints and bboxes
            gt_keypoints = torch.stack(
                [
                    torch.tensor(
                        gt.keypoints, dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad
                    )
                    for gt in gts
                ]
            )  # [num_gts, 3*k]
            gt_bboxes = torch.stack(
                [
                    torch.tensor(gt.bbox, dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad)
                    for gt in gts
                ]
            )  # [num_gts, 4]
            gt_areas = torch.tensor(
                [gt.area for gt in gts], dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad
            )

            # Stack all dt keypoints
            dt_keypoints = torch.stack(
                [
                    torch.tensor(
                        dt.keypoints, dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad
                    )
                    for dt in dts
                ]
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
                / (gt_areas.unsqueeze(0).unsqueeze(-1) + torch.finfo(self.float_dtype).eps)
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
            denom_counts = torch.where(
                k1_pos,
                k1.to(dtype=self.float_dtype),
                torch.full_like(k1, float(k), dtype=self.float_dtype),
            )  # [G]
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

        gt_ig = torch.tensor(ignored, dtype=torch.bool, device=self.device, requires_grad=self.requires_grad)
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
        _limit = float(1.0 - torch.finfo(self.float_dtype).eps)
        iou_thrs_t = torch.tensor(
            [min(float(t.item()), _limit) for t in self.params.iouThrs],
            device=self.device,
            dtype=self.float_dtype,
        )  # [T]

        num_iou_thrs = len(self.params.iouThrs)  # T
        num_gt = len(gt)  # G
        num_dt = len(dt)  # D
        gtm = torch.zeros((num_iou_thrs, num_gt), dtype=torch.long, device=self.device)
        dtm = torch.zeros((num_iou_thrs, num_dt), dtype=torch.long, device=self.device)
        dt_ig = torch.zeros((num_iou_thrs, num_dt), dtype=torch.bool, device=self.device)
        if len(ious) != 0:
            # --- Vectorized matching: iterate over D detections, vectorize over T thresholds ---
            # GTs are already sorted: non-ignored (gt_ig==0) first, ignored (gt_ig==1) last.
            iscrowd_bool = iscrowd_t.bool()  # [G]
            gt_ig_bool = gt_ig.bool()  # [G]
            non_ignored = ~gt_ig_bool  # [G]

            # Pre-extract GT and DT ids as tensors for fast assignment
            gt_ids_t = torch.tensor([g.id for g in gt], dtype=torch.long, device=self.device)  # [G]
            dt_ids_t = torch.tensor([d.id for d in dt], dtype=torch.long, device=self.device)  # [D]

            # ious: [D, G] already (subset-selected above)
            ious_f = ious.to(dtype=self.float_dtype, device=self.device)  # [D, G]

            # matched_gt_t[t, g] = True when GT g has been matched at threshold t
            matched_gt_t = torch.zeros((num_iou_thrs, num_gt), dtype=torch.bool, device=self.device)

            # Greedy matching per detection (sequential over D), vectorized over T:
            for dind in range(num_dt):
                iou_d = ious_f[dind]  # [G] - IoU of this detection against all GTs

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
            dtIds=torch.tensor([d.id for d in dt], dtype=torch.long, device=self.device),
            gtIds=torch.tensor([g.id for g in gt], dtype=torch.long, device=self.device),
            dtMatches=dtm,
            gtMatches=gtm,
            dtScores=torch.tensor(
                [d.score for d in dt], dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad
            ),
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
        if self._state is None:
            self.logger.info("Please run evaluate() first")
            return
        s = self._state
        # allows input customized parameters
        p = p or self.params
        dev, fdtype = self.device, self.float_dtype

        p.catIds = p.catIds if p.useCats == 1 else [-1]
        num_iou_thrs = len(p.iouThrs)  # T
        num_rec_thrs = len(p.recThrs)  # R
        num_cat_ids = len(p.catIds) if p.useCats else 1  # K
        num_area_rng = len(p.areaRng)  # A
        num_max_dets = len(p.maxDets)  # M
        precision = torch.full(
            (num_iou_thrs, num_rec_thrs, num_cat_ids, num_area_rng, num_max_dets), -1.0, dtype=fdtype, device=dev
        )  # -1 for the precision of absent categories
        recall = torch.full((num_iou_thrs, num_cat_ids, num_area_rng, num_max_dets), -1.0, dtype=fdtype, device=dev)
        scores = torch.full_like(precision, -1.0)

        # list positions are used as indices into the evaluated layout, exactly like the reference
        set_cat_ids = set(s.cat_ids)
        set_area_rng = set(map(tuple, self._paramsEval.areaRng))
        set_max_dets = set(self._paramsEval.maxDets)
        set_img_ids = set(self._paramsEval.imgIds)
        cat_ids_indices = [n for n, k in enumerate(p.catIds) if k in set_cat_ids]
        max_dets_indices = [m for m in p.maxDets if m in set_max_dets]
        area_rng_indices = [n for n, a in enumerate(tuple(x) for x in p.areaRng) if a in set_area_rng]
        img_ids_indices = [n for n, i in enumerate(p.imgIds) if i in set_img_ids]

        cat_map = torch.full((len(s.cat_ids),), -1, dtype=torch.long)
        for k, k0 in enumerate(cat_ids_indices):
            if k0 < len(s.cat_ids):
                cat_map[k0] = k
        img_sel = torch.zeros(len(s.img_ids), dtype=torch.bool)
        img_sel[[i for i in img_ids_indices if i < len(s.img_ids)]] = True
        pair_cat = cat_map[torch.tensor(s.pair_cat, dtype=torch.long)]
        pair_cat = torch.where(img_sel[torch.tensor(s.pair_img, dtype=torch.long)], pair_cat, -1).to(dev)
        pair_img = torch.tensor(s.pair_img, dtype=torch.long, device=dev)

        a_idx = torch.tensor(area_rng_indices, dtype=torch.long, device=dev)
        gt_cat = pair_cat[s.gt_pair]
        counted = ~s.gt_ignore[a_idx] & (gt_cat >= 0)
        npig = torch.zeros((a_idx.numel(), num_cat_ids), dtype=torch.long, device=dev)
        npig.scatter_add_(1, gt_cat.clamp(min=0).expand_as(counted), counted.long())

        prec, rec, sc = self._accumulate(
            dt_cat=pair_cat[s.dt_pair],
            dt_order_key=pair_img[s.dt_pair] * (s.max_det + 1) + s.dt_rank,
            dt_rank=s.dt_rank,
            dt_score=s.dt_scores,
            dt_tp=s.dt_tp[a_idx],
            dt_ignore=s.dt_ignore[a_idx],
            npig=npig,
            max_dets=max_dets_indices,
            rec_thrs=p.recThrs.to(device=dev, dtype=fdtype),
            num_cats=num_cat_ids,
        )
        num_a, num_m = a_idx.numel(), len(max_dets_indices)
        precision[:, :, :, :num_a, :num_m] = prec
        recall[:, :, :num_a, :num_m] = rec
        scores[:, :, :, :num_a, :num_m] = sc
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
                    iou_thr_tensor = torch.tensor(
                        iouThr, dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad
                    )
                    thr = torch.where(
                        iou_thr_tensor == self.params.iouThrs.to(device=self.device, dtype=self.float_dtype)
                    )[0]
                    prec = prec[thr]
                prec = prec[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                prec = self.eval.recall
                if iouThr is not None:
                    iou_thr_tensor = torch.tensor(
                        iouThr, dtype=self.float_dtype, device=self.device, requires_grad=self.requires_grad
                    )
                    thr = torch.where(
                        iou_thr_tensor == self.params.iouThrs.to(device=self.device, dtype=self.float_dtype)
                    )[0]
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
            stats = torch.zeros((12,), dtype=self.float_dtype, device=self.device)
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
            stats = torch.zeros((10,), dtype=self.float_dtype, device=self.device)
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
