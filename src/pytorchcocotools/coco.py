from __future__ import annotations

from collections import defaultdict
import copy
import dataclasses
import itertools
import json
import logging
from pathlib import Path
import time
from typing import cast

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools import mask
from pytorchcocotools.internal.entities import RleObj, RleObjs
from pytorchcocotools.internal.structure import CocoDetectionDataset
from pytorchcocotools.internal.structure.additional import ResultAnnotation
from pytorchcocotools.internal.structure.annotations import (
    CocoAnnotationDetection,
    CocoAnnotationKeypointDetection,
    CocoAnnotationObjectDetection,
)
from pytorchcocotools.internal.structure.categories import (
    CocoCategoriesDetection,
    CocoCategoriesKeypointDetection,
    CocoCategoriesObjectDetection,
)
from pytorchcocotools.internal.structure.images import CocoImage
from pytorchcocotools.utils.logging import get_logger


class COCO:
    def __init__(self, annotation_file: str | None = None, *, enable_logging: bool = True) -> None:
        """Constructor of Microsoft COCO helper class for reading and visualizing annotations.

        Args:
            annotation_file: The location of annotation file. Defaults to None.
            enable_logging: Whether to enable logging. Defaults to True.
        """
        self.anns: dict[int, CocoAnnotationDetection] = {}
        self.cats: dict[int, CocoCategoriesDetection] = {}
        self.imgs: dict[int, CocoImage] = {}
        self.imgToAnns: defaultdict[int, list[CocoAnnotationDetection]] = defaultdict(list[CocoAnnotationDetection])
        self.catToImgs: defaultdict[int, list[int]] = defaultdict(list[int])
        self.logger = get_logger(self.__class__.__name__) if enable_logging else logging.getLogger(__name__)
        if annotation_file is not None:
            self.logger.info("loading annotations into memory...")
            tic = time.time()
            path = Path(annotation_file)
            with path.open("r") as file:
                dataset: CocoDetectionDataset = CocoDetectionDataset.from_dict(json.load(file))
            assert isinstance(dataset, CocoDetectionDataset), f"Annotation file format {type(dataset)} not supported."  # noqa: S101 # nosec
            self.logger.info(f"Done (t={time.time() - tic:0.2f}s)")
            self.dataset = dataset
            self.createIndex()
        else:
            self.logger.debug("Annotation file not provided, creating empty dataset.")

    def createIndex(self) -> None:  # noqa: N802
        self.logger.info("creating index...")
        anns: dict[int, CocoAnnotationDetection] = {}
        cats: dict[int, CocoCategoriesDetection] = {}
        imgs: dict[int, CocoImage] = {}
        img_to_anns: defaultdict[int, list[CocoAnnotationDetection]] = defaultdict(list[CocoAnnotationDetection])
        cat_to_imgs: defaultdict[int, list[int]] = defaultdict(list[int])
        for ann in self.dataset.annotations:
            img_to_anns[ann.image_id].append(ann)
            anns[ann.id] = ann

        for img in self.dataset.images:
            imgs[img.id] = img

        for cat in self.dataset.categories:
            cats[cat.id] = cat

        for ann in self.dataset.annotations:
            cat_to_imgs[ann.category_id].append(ann.image_id)

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

        self.logger.info("index created!")

    def info(self) -> None:
        """Print information about the annotation file."""
        for key, value in self.dataset.info.items():
            self.logger.info(f"{key}: {value}")

    def getAnnIds(  # noqa: N802
        self,
        imgIds: int | list[int] | None = None,  # noqa: N803
        catIds: int | list[int] | None = None,  # noqa: N803
        areaRng: float | list[float] | None = None,  # noqa: N803
        iscrowd: bool | None = None,
    ) -> list[int]:
        """Get ann ids that satisfy given filter conditions. default skips that filter.

        Args:
            imgIds: Get anns for given imgs.
            catIds: Get anns for given cats.
            areaRng: get anns for given area range (e.g. [0 inf]).
            iscrowd: get anns for given crowd label (False or True).

        Returns:
            Integer array of ann ids.
        """
        img_ids: list = imgIds if isinstance(imgIds, list) else [imgIds] if imgIds else []
        cat_ids: list = catIds if isinstance(catIds, list) else [catIds] if catIds else []
        area_rng: list = areaRng if isinstance(areaRng, list) else [areaRng] if areaRng else []

        if len(img_ids) == len(cat_ids) == len(area_rng) == 0:
            anns = self.dataset.annotations
        else:
            if img_ids:
                lists = [self.imgToAnns[imgId] for imgId in img_ids if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset.annotations
            anns = [ann for ann in anns if ann.category_id in cat_ids] if cat_ids else anns
            anns = [ann for ann in anns if ann.area > area_rng[0] and ann.area < area_rng[1]] if area_rng else anns
        return [ann.id for ann in anns if ann.iscrowd == iscrowd] if iscrowd is not None else [ann.id for ann in anns]

    def getCatIds(  # noqa: N803, N802
        self,
        catNms: str | list[str] | None = None,  # noqa: N803, N802
        supNms: str | list[str] | None = None,  # noqa: N803, N802
        catIds: int | list[int] | None = None,  # noqa: N803, N802
    ) -> list[int]:
        """Filtering parameters. default skips that filter.

        Args:
            catNms: Get cats for given cat names.
            supNms: Get cats for given supercategory names.
            catIds: Get cats for given cat ids.

        Returns:
            Integer array of cat ids.
        """
        cat_nms: list = catNms if isinstance(catNms, list) else [catNms] if catNms else []
        sup_nms: list = supNms if isinstance(supNms, list) else [supNms] if supNms else []
        cat_ids: list = catIds if isinstance(catIds, list) else [catIds] if catIds else []

        cats = self.dataset.categories
        if not len(cat_nms) == len(sup_nms) == len(cat_ids) == 0:
            cats = [cat for cat in cats if cat.name in cat_nms] if cat_nms else cats
            cats = [cat for cat in cats if cat.supercategory in sup_nms] if sup_nms else cats
            cats = [cat for cat in cats if cat.id in cat_ids] if cat_ids else cats
        return [cat.id for cat in cats]

    def getImgIds(self, imgIds: int | list[int] | None = None, catIds: int | list[int] | None = None) -> list[int]:  # noqa: N802, N803
        """Get img ids that satisfy given filter conditions.

        Args:
            imgIds: Get imgs for given ids.
            catIds: Get imgs with all given cats.

        Returns:
            Integer array of img ids.
        """
        img_ids: list = imgIds if isinstance(imgIds, list) else [imgIds] if imgIds else []
        cat_ids: list = catIds if isinstance(catIds, list) else [catIds] if catIds else []

        if len(img_ids) == len(cat_ids) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(img_ids)
            for i, cat_id in enumerate(cat_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[cat_id])
                else:
                    ids &= set(self.catToImgs[cat_id])
        return list(ids)

    def loadAnns(self, ids: int | list[int] | None = None) -> list[CocoAnnotationDetection]:  # noqa: N802
        """Load anns with the specified ids.

        Args:
            ids: Integer ids specifying anns.

        Returns:
            Loaded ann objects.
        """
        ann_ids: int | list[int] = ids if ids else []
        if isinstance(ann_ids, list):
            return [self.anns[id] for id in ann_ids]
        elif isinstance(ann_ids, int):
            return [self.anns[ann_ids]]

    def loadCats(self, ids: int | list[int] | None = None) -> list[CocoCategoriesDetection]:  # noqa: N802
        """Load cats with the specified ids.

        Args:
            ids: Integer ids specifying cats.

        Returns:
            Loaded cat objects.
        """
        cat_ids: int | list[int] = ids if ids else []
        if isinstance(cat_ids, list):
            return [self.cats[id] for id in cat_ids]
        elif isinstance(cat_ids, int):
            return [self.cats[cat_ids]]

    def loadImgs(self, ids: int | list[int] | None = None) -> list[CocoImage]:  # noqa: N802
        """Load anns with the specified ids.

        Args:
            ids: Integer ids specifying img

        Returns:
            Loaded img objects.
        """
        img_ids: int | list[int] = ids if ids else []
        if isinstance(img_ids, list):
            return [self.imgs[id] for id in img_ids]
        elif isinstance(img_ids, int):
            return [self.imgs[img_ids]]

    def showAnns(self, anns: list[CocoAnnotationDetection], draw_bbox: bool = False) -> None:  # noqa: N802
        """Display the specified annotations.

        Args:
            anns: Annotations to display.
            draw_bbox: Whether to draw the bounding boxes or not.

        Raises:
            Exception: _description_

        Returns:
            _description_
        """
        annotations = (
            [CocoDetectionDataset._get_annotation(ann) for ann in anns]
            if isinstance(anns, list) and all(isinstance(ann, dict) for ann in anns)
            else anns
        )
        if len(annotations) == 0:
            return
        if "segmentation" in annotations[0] or "keypoints" in annotations[0]:
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in annotations:
                c = (torch.rand((1, 3)) * 0.6 + 0.4).tolist()[0]
                if "segmentation" in ann:  # isinstance(ann, CocoAnnotationObjectDetection)
                    ann = cast(CocoAnnotationObjectDetection, ann)
                    if isinstance(ann.segmentation, list):
                        # polygon
                        for seg in ann.segmentation:
                            poly = torch.Tensor(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann.image_id]
                        if isinstance(ann.segmentation["counts"], list):
                            rle = mask.frPyObjects(
                                [cast(RleObj, ann.segmentation)], t.height, t.width
                            )  # TODO: uncompressed, type hint not perfect
                        else:
                            rle = cast(RleObjs, [ann.segmentation])
                        m = mask.decode(rle)
                        img = torch.ones((m.shape[0], m.shape[1], 3))
                        color_mask = torch.tensor([2.0, 166.0, 101.0]) / 255 if ann.iscrowd else torch.rand((1, 3))[0]
                        for i in range(3):
                            img[:, :, i] = float(color_mask[i])
                        ax.imshow(torch.dstack((img, m * 0.5)))
                if "keypoints" in ann and isinstance(ann["keypoints"], list):
                    # turn skeleton into zero-based index
                    sks = torch.Tensor(self.loadCats(ann.category_id)[0]["skeleton"]) - 1
                    kp = torch.Tensor(ann["keypoints"])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if torch.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(
                        x[v > 0], y[v > 0], "o", markersize=8, markerfacecolor=c, markeredgecolor="k", markeredgewidth=2
                    )
                    plt.plot(
                        x[v > 1], y[v > 1], "o", markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2
                    )

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann.bbox
                    poly = [
                        [bbox_x, bbox_y],
                        [bbox_x, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y],
                    ]
                    np_poly = torch.Tensor(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
            ax.add_collection(p)

    def loadPyTorchAnnotations(self, data: torch.Tensor) -> list[ResultAnnotation]:  # noqa: N802
        """Convert result data from a torch Tensor [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}.

        Args:
            data: _description_

        Returns:
            _description_
        """
        self.logger.info("Converting ndarray to lists...")
        assert isinstance(data, torch.Tensor)  # noqa: S101 # nosec
        self.logger.debug(data.shape)
        assert data.shape[1] == 7  # noqa: S101 # nosec
        n = data.shape[0]
        ann = []
        for i in range(n):
            if i % 1000000 == 0:
                self.logger.info(f"{i}/{n}")
            ann += [
                ResultAnnotation(
                    image_id=int(data[i, 0]),
                    bbox=[data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                    score=data[i, 5],
                    category_id=int(data[i, 6]),
                )
            ]
        return ann

    def loadRes(self, resFile: str | Tensor | list[ResultAnnotation | CocoAnnotationDetection]) -> COCO:  # noqa: N802, N803
        """Load result file and return a result api object.

        Args:
            resFile: File name of result file.

        Returns:
            The result api object.
        """
        res = COCO()
        res.dataset.info = copy.deepcopy(self.dataset.info)
        res.dataset.images = list(self.dataset.images)

        self.logger.info("Loading and preparing results...")
        tic = time.time()
        if isinstance(resFile, str):
            with Path(resFile).open("r") as file:
                anns = json.load(file)
                anns = [CocoDetectionDataset._get_annotation(ann) for ann in anns]
        elif isinstance(resFile, torch.Tensor):
            self.logger.warning("Converting torch.Tensor to list of ResultAnnotation")
            anns = self.loadPyTorchAnnotations(resFile)  # TODO: is gonna cause issues
        else:
            anns = resFile

        assert isinstance(anns, list), "results in not an array of objects"  # noqa: S101 # nosec

        anns_img_ids = [ann.image_id for ann in anns]
        assert set(anns_img_ids) == (  # noqa: S101 # nosec
            set(anns_img_ids) & set(self.getImgIds())
        ), "Results do not correspond to current coco set"

        new_anns = []
        for id, ann in enumerate(anns):
            if "bbox" in ann and ann["bbox"] != []:
                new_ann = CocoAnnotationObjectDetection(**dataclasses.asdict(ann))  # pyright: ignore [reportArgumentType]
                bb = ann.bbox
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                new_ann.segmentation = ann.get("segmentation", [[x1, y1, x1, y2, x2, y2, x2, y1]])
                new_ann.area = bb[2] * bb[3]
                new_ann.id = id + 1
                new_ann.iscrowd = False
                new_anns.append(new_ann)
            elif "segmentation" in ann:
                new_ann = CocoAnnotationObjectDetection(**dataclasses.asdict(ann))  # pyright: ignore [reportArgumentType]
                # now only support compressed RLE format as segmentation results
                new_ann.area = float(mask.area(cast(RleObjs, ann.segmentation))[0])  # pyright: ignore[reportAttributeAccessIssue]
                if "bbox" not in ann:
                    new_ann.bbox = mask.toBbox(cast(RleObjs, ann.segmentation))  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    new_ann.bbox = ann.bbox
                new_ann.id = id + 1
                new_ann.iscrowd = False
                new_anns.append(new_ann)
            elif "keypoints" in ann:
                new_ann = CocoAnnotationKeypointDetection(**dataclasses.asdict(ann))  # pyright: ignore [reportArgumentType]
                # keypoints
                s = ann["keypoints"]
                x = s[::3]
                y = s[1::3]
                x0, x1, y0, y1 = min(x), max(x), min(y), max(y)
                new_ann.area = float((x1 - x0) * (y1 - y0))
                new_ann.id = id + 1
                new_ann.bbox = [x0, y0, x1 - x0, y1 - y0]
                new_anns.append(new_ann)
        self.logger.info(f"DONE (t={time.time() - tic:0.2f}s)")

        res.dataset.categories = copy.deepcopy(self.dataset.categories)
        res.dataset.annotations = new_anns
        res.createIndex()
        return res

    def download(self, tarDir: str | None = None, imgIds: list[int] | None = None) -> None:  # noqa: N803
        """Download COCO images from mscoco.org server.

        Args:
            tarDir: COCO results directory name.
            imgIds: Images to be downloaded.
        """
        from urllib.request import urlretrieve

        imgIds = imgIds if imgIds else []  # noqa: N806
        target_dir = Path(tarDir or ".")
        if target_dir is None:
            self.logger.info("Please specify target directory")
            return
        imgs = self.imgs.values() if len(imgIds) == 0 else self.loadImgs(imgIds)
        num_imgs = len(imgs)  # N
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = target_dir / img.file_name
            if not Path.exists(fname):
                urlretrieve(img.coco_url, fname)  # noqa: S310 # nosec
            self.logger.info(f"downloaded {i}/{num_imgs} images (t={time.time() - tic:0.1f}s)")

    def annToRLE(self, ann: CocoAnnotationDetection) -> RleObjs | RleObj:  # noqa: N802
        """Convert annotation which can be polygons, uncompressed RLE to RLE.

        Args:
            ann: _description_

        Returns:
            _description_
        """
        t = self.imgs[ann.image_id]
        h, w = t.height, t.width
        segm = ann.segmentation
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = cast(RleObjs, mask.frPyObjects(cast(list[list[float]], segm), h, w))
            merged = mask.merge(rles)
            return merged
        elif isinstance(segm, dict) and isinstance(segm["counts"], list):
            # uncompressed RLE
            objs = mask.frPyObjects(cast(RleObj, segm), h, w)
            return objs
        else:
            return segm  # type: ignore  # noqa: PGH003 # TODO: fix this, shouldn't happen???

    def annToMask(self, ann: dict) -> tv.Mask:  # noqa: N802
        """Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.

        Args:
            ann: _description_

        Returns:
            _description_
        """
        rle = self.annToRLE(CocoDetectionDataset._get_annotation(ann))
        decoded = mask.decode(rle)
        return decoded
