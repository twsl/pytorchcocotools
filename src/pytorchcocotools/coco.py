# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

from collections import defaultdict
import copy
import itertools
import json
from logging import Logger
from pathlib import Path
import time
from urllib.request import urlretrieve

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from pytorchcocotools import mask, utils
import torch
from torch import Tensor


def _isArrayLike(obj):  # noqa: N802
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO:
    logger: Logger

    def __init__(self, annotation_file: str = None) -> None:
        """Constructor of Microsoft COCO helper class for reading and visualizing annotations.

        Args:
            annotation_file: The location of annotation file. Defaults to None.
        """
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.logger = utils.get_logger(__name__)
        if annotation_file is not None:
            self.logger.info("loading annotations into memory...")
            tic = time.time()
            path = Path(annotation_file)
            with path.open("r") as file:
                dataset = json.load(file)
            assert isinstance(dataset, dict), f"annotation file format {type(dataset)} not supported"  # noqa: S101
            self.logger.info(f"Done (t={time.time() - tic:0.2f}s)")
            self.dataset = dataset
            self.createIndex()

    def createIndex(self) -> None:  # noqa: N802
        self.logger.info("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns = defaultdict(list)  # noqa: N806
        catToImgs = defaultdict(list)  # noqa: N806
        for ann in self.dataset.get("annotations", []):
            imgToAnns[ann["image_id"]].append(ann)
            anns[ann["id"]] = ann

        for img in self.dataset.get("images", []):
            imgs[img["id"]] = img

        for cat in self.dataset.get("categories", []):
            cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        self.logger.info("index created!")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self) -> None:
        """Print information about the annotation file."""
        for key, value in self.dataset["info"].items():
            self.logger.info(f"{key}: {value}")

    def getAnnIds(  # noqa: N802
        self,
        imgIds: int | list[int] = None,  # noqa: N803
        catIds: int | list[int] = None,  # noqa: N803
        areaRng: float | list[float] = None,  # noqa: N803
        iscrowd: bool = None,
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
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds] if imgIds else []  # noqa: N806
        catIds = catIds if _isArrayLike(catIds) else [catIds] if catIds else []  # noqa: N806
        areaRng = areaRng if _isArrayLike(areaRng) else [areaRng] if areaRng else []  # noqa: N806

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            if imgIds:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = [ann for ann in anns if ann["category_id"] in catIds] if catIds else anns
            anns = [ann for ann in anns if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]] if areaRng else anns
        return (
            [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
            if iscrowd is not None
            else [ann["id"] for ann in anns]
        )

    def getCatIds(  # noqa: N803, N802
        self,
        catNms: str | list[str] = None,  # noqa: N803, N802
        supNms: str | list[str] = None,  # noqa: N803, N802
        catIds: int | list[int] = None,  # noqa: N803, N802
    ) -> list[int]:  # noqa: N803, N802
        """Filtering parameters. default skips that filter.

        Args:
            catNms: Get cats for given cat names.
            supNms: Get cats for given supercategory names.
            catIds: Get cats for given cat ids.

        Returns:
            Integer array of cat ids.
        """
        catIds = catIds if catIds else []  # noqa: N806
        catNms = catNms if _isArrayLike(catNms) else [catNms] if catNms else []  # noqa: N806
        supNms = supNms if _isArrayLike(supNms) else [supNms] if supNms else []  # noqa: N806
        catIds = catIds if _isArrayLike(catIds) else [catIds] if catIds else []  # noqa: N806

        cats = self.dataset["categories"]
        if not len(catNms) == len(supNms) == len(catIds) == 0:
            cats = [cat for cat in cats if cat["name"] in catNms] if catNms else cats
            cats = [cat for cat in cats if cat["supercategory"] in supNms] if supNms else cats
            cats = [cat for cat in cats if cat["id"] in catIds] if catIds else cats
        return [cat["id"] for cat in cats]

    def getImgIds(self, imgIds: int | list[int] = None, catIds: int | list[int] = None) -> list[int]:  # noqa: N802, N803
        """Get img ids that satisfy given filter conditions.

        Args:
            imgIds: Get imgs for given ids.
            catIds: Get imgs with all given cats.

        Returns:
            Integer array of img ids.
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds] if imgIds else []  # noqa: N806
        catIds = catIds if _isArrayLike(catIds) else [catIds] if catIds else []  # noqa: N806

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):  # noqa: N806
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids: int | list[int] = None) -> list:  # noqa: N802
        """Load anns with the specified ids.

        Args:
            ids: Integer ids specifying anns.

        Returns:
            Loaded ann objects.
        """
        ids = ids if ids else []
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif isinstance(ids, int):
            return [self.anns[ids]]

    def loadCats(self, ids: int | list[int] = None) -> list:  # noqa: N802
        """Load cats with the specified ids.

        Args:
            ids: Integer ids specifying cats.

        Returns:
            Loaded cat objects.
        """
        ids = ids if ids else []
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif isinstance(ids, int):
            return [self.cats[ids]]

    def loadImgs(self, ids: int | list[int] = None) -> list:  # noqa: N802
        """Load anns with the specified ids.

        Args:
            ids: Integer ids specifying img

        Returns:
            Loaded img objects.
        """
        ids = ids if ids else []
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif isinstance(ids, int):
            return [self.imgs[ids]]

    def showAnns(self, anns: list, draw_bbox: bool = False) -> None:  # noqa: N802
        """Display the specified annotations.

        Args:
            anns: Annotations to display.
            draw_bbox: Whether to draw the bounding boxes or not.

        Raises:
            Exception: _description_

        Returns:
            _description_
        """
        if len(anns) == 0:
            return 0
        if "segmentation" in anns[0] or "keypoints" in anns[0]:
            dataset_type = "instances"
        elif "caption" in anns[0]:
            dataset_type = "captions"
        else:
            raise Exception("datasetType not supported")  # noqa: TRY002
        if dataset_type == "instances":
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (torch.rand((1, 3)) * 0.6 + 0.4).tolist()[0]
                if "segmentation" in ann:
                    if isinstance(ann["segmentation"], list):
                        # polygon
                        for seg in ann["segmentation"]:
                            poly = torch.Tensor(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann["image_id"]]
                        if isinstance(ann["segmentation"]["counts"], list):
                            rle = mask.frPyObjects([ann["segmentation"]], t["height"], t["width"])
                        else:
                            rle = [ann["segmentation"]]
                        m = mask.decode(rle)
                        img = torch.ones((m.shape[0], m.shape[1], 3))
                        if ann["iscrowd"] == 1:
                            color_mask = torch.array([2.0, 166.0, 101.0]) / 255
                        if ann["iscrowd"] == 0:
                            color_mask = torch.rand((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(torch.dstack((img, m * 0.5)))
                if "keypoints" in ann and isinstance(ann["keypoints"], list):
                    # turn skeleton into zero-based index
                    sks = torch.Tensor(self.loadCats(ann["category_id"])[0]["skeleton"]) - 1
                    kp = torch.Tensor(ann["keypoints"])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if torch.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(
                        x[v > 0],
                        y[v > 0],
                        "o",
                        markersize=8,
                        markerfacecolor=c,
                        markeredgecolor="k",
                        markeredgewidth=2,
                    )
                    plt.plot(
                        x[v > 1],
                        y[v > 1],
                        "o",
                        markersize=8,
                        markerfacecolor=c,
                        markeredgecolor=c,
                        markeredgewidth=2,
                    )

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann["bbox"]
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
        elif dataset_type == "captions":
            for ann in anns:
                self.logger.info(ann["caption"])

    def loadRes(self, resFile: str | Tensor) -> "COCO":  # noqa: N802, N803
        """Load result file and return a result api object.

        Args:
            resFile: File name of result file.

        Returns:
            The result api object.
        """
        res = COCO()
        res.dataset["images"] = list(self.dataset["images"])

        self.logger.info("Loading and preparing results...")
        tic = time.time()
        if isinstance(resFile, str):
            path = Path(resFile)
            with path.open("r") as file:
                anns = json.load(file)
        elif type(resFile) == torch.Tensor:
            anns = self.loadPyTorchAnnotations(resFile)
        else:
            anns = resFile
        assert isinstance(anns, list), "results in not an array of objects"  # noqa: S101
        anns_img_ids = [ann["image_id"] for ann in anns]
        assert set(anns_img_ids) == (  # noqa: S101
            set(anns_img_ids) & set(self.getImgIds())
        ), "Results do not correspond to current coco set"
        if "caption" in anns[0]:
            img_ids = {img["id"] for img in res.dataset["images"]} & {ann["image_id"] for ann in anns}
            res.dataset["images"] = [img for img in res.dataset["images"] if img["id"] in img_ids]
            for id, ann in enumerate(anns):
                ann["id"] = id + 1
        elif "bbox" in anns[0] and anns[0]["bbox"] != []:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                bb = ann["bbox"]
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if "segmentation" not in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = mask.area(ann["segmentation"])
                if "bbox" not in ann:
                    ann["bbox"] = mask.toBbox(ann["segmentation"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                s = ann["keypoints"]
                x = s[::3]
                y = s[1::3]
                x0, x1, y0, y1 = torch.min(x), torch.max(x), torch.min(y), torch.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = id + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        self.logger.info(f"DONE (t={time.time() - tic:0.2f}s)")

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def download(self, tarDir: str = None, imgIds: list[int] = None) -> None:  # noqa: N803
        """Download COCO images from mscoco.org server.

        Args:
            tarDir: COCO results directory name.
            imgIds: Images to be downloaded.

        Returns:
            _description_
        """
        imgIds = imgIds if imgIds else []  # noqa: N806
        if tarDir is None:
            self.logger.info("Please specify target directory")
            return -1
        imgs = self.imgs.values() if len(imgIds) == 0 else self.loadImgs(imgIds)
        num_imgs = len(imgs)  # N
        if not Path.exists(tarDir):
            Path.mkdir(tarDir, parents=True)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = Path(tarDir) / img["file_name"]
            if not Path.exists(fname):
                urlretrieve(img["coco_url"], fname)  # noqa: S310
            self.logger.info(f"downloaded {i}/{num_imgs} images (t={time.time() - tic:0.1f}s)")

    def loadPyTorchAnnotations(self, data: torch.Tensor) -> list[list[dict]]:  # noqa: N802
        """Convert result data from a torch Tensor [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}.

        Args:
            data: _description_

        Returns:
            _description_
        """
        self.logger.info("Converting ndarray to lists...")
        assert type(data) == torch.Tensor  # noqa: S101
        self.logger.info(data.shape)
        assert data.shape[1] == 7  # noqa: S101
        n = data.shape[0]
        ann = []
        for i in range(n):
            if i % 1000000 == 0:
                self.logger.info(f"{i}/{n}")
            ann += [
                {
                    "image_id": int(data[i, 0]),
                    "bbox": [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                    "score": data[i, 5],
                    "category_id": int(data[i, 6]),
                }
            ]
        return ann

    def annToRLE(self, ann: dict) -> dict | list[dict]:  # noqa: N802
        """Convert annotation which can be polygons, uncompressed RLE to RLE.

        Args:
            ann: _description_

        Returns:
            _description_
        """
        t = self.imgs[ann["image_id"]]
        h, w = t["height"], t["width"]
        segm = ann["segmentation"]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask.frPyObjects(segm, h, w)
            return mask.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            return mask.frPyObjects(segm, h, w)
        else:
            return segm

    def annToMask(self, ann: dict) -> Tensor:  # noqa: N802
        """Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.

        Args:
            ann: _description_

        Returns:
            _description_
        """
        rle = self.annToRLE(ann)
        return mask.decode(rle)
