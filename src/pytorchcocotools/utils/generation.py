import math
import random
from typing import Any

from pytorchcocotools.internal.entities import IoUType

NUM_IMAGES = 500
NUM_CATEGORIES = 10
IMG_W, IMG_H = 640, 480
NUM_KEYPOINTS = 17


def _polygon(x: float, y: float, w: float, h: float, rng: random.Random) -> list[float]:
    cx, cy = x + w / 2, y + h / 2
    num_vertices = rng.randint(6, 16)
    poly: list[float] = []
    for i in range(num_vertices):
        angle = 2 * math.pi * i / num_vertices
        r = rng.uniform(0.6, 1.0)
        poly += [cx + r * w / 2 * math.cos(angle), cy + r * h / 2 * math.sin(angle)]
    return poly


def _box(rng: random.Random, image_width: int, image_height: int) -> list[float]:
    w = rng.uniform(min(4, image_width), min(300, image_width))
    h = rng.uniform(min(4, image_height), min(250, image_height))
    return [rng.uniform(0, image_width - w), rng.uniform(0, image_height - h), w, h]


def _jitter(box: list[float], rng: random.Random, image_width: int, image_height: int) -> list[float]:
    x, y, w, h = box
    s = 0.15
    w2 = min(image_width, max(1.0, w * rng.uniform(1 - s, 1 + s)))
    h2 = min(image_height, max(1.0, h * rng.uniform(1 - s, 1 + s)))
    x2 = min(max(0.0, x + w * rng.uniform(-s, s)), image_width - w2)
    y2 = min(max(0.0, y + h * rng.uniform(-s, s)), image_height - h2)
    return [x2, y2, w2, h2]


def _keypoints(
    box: list[float], rng: random.Random, visible_prob: float, num_keypoints: int
) -> tuple[list[float], int]:
    x, y, w, h = box
    kps: list[float] = []
    num_visible = 0
    for _ in range(num_keypoints):
        v = 2 if rng.random() < visible_prob else 0
        num_visible += v > 0
        kps += [x + rng.uniform(0, w), y + rng.uniform(0, h), v]
    return kps, num_visible


def _detection(
    image_id: int,
    category_id: int,
    box: list[float],
    iou_type: IoUType,
    rng: random.Random,
    num_keypoints: int,
) -> dict:
    det: dict[str, Any] = {"image_id": image_id, "category_id": category_id, "score": round(rng.random(), 4)}
    if iou_type == "bbox":
        det["bbox"] = box
    elif iou_type == "segm":
        det["segmentation"] = [_polygon(*box, rng)]
        det["bbox"] = box
    else:
        det["keypoints"], _ = _keypoints(box, rng, 1.0, num_keypoints)
    return det


def generate_mock_data(
    iou_type: IoUType,
    seed: int = 0,
    *,
    num_images: int = NUM_IMAGES,
    num_categories: int = NUM_CATEGORIES,
    image_width: int = IMG_W,
    image_height: int = IMG_H,
    num_keypoints: int = NUM_KEYPOINTS,
    annotations_per_image: tuple[int, int] = (0, 25),
    unmatched_detections_per_image: tuple[int, int] = (0, 50),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate a synthetic COCO dataset and matching detections.

    Args:
        iou_type: Evaluation type used to shape annotations and detections.
        seed: Random seed for repeatable output.
        num_images: Number of images to generate.
        num_categories: Number of categories to generate.
        image_width: Width of each generated image.
        image_height: Height of each generated image.
        num_keypoints: Number of keypoints per person for keypoint evaluation.
        annotations_per_image: Inclusive minimum and maximum ground-truth annotations per image.
        unmatched_detections_per_image: Inclusive minimum and maximum extra detections per image.

    Returns:
        A COCO dataset dictionary and a list of detection dictionaries.
    """
    if num_images < 0 or num_categories < 1 or image_width < 1 or image_height < 1:
        raise ValueError("num_images must be nonnegative; categories and image dimensions must be positive")
    if iou_type == "keypoints" and num_keypoints < 1:
        raise ValueError("num_keypoints must be positive for keypoint evaluation")
    if any(low < 0 or high < low for low, high in (annotations_per_image, unmatched_detections_per_image)):
        raise ValueError("per-image count ranges must have nonnegative bounds in ascending order")

    rng = random.Random(seed)  # noqa: S311
    images: list[dict[str, Any]] = [
        {"id": i, "width": image_width, "height": image_height, "file_name": f"{i}.jpg"}
        for i in range(1, num_images + 1)
    ]
    categories: list[dict[str, Any]]
    if iou_type == "keypoints":
        categories = [
            {
                "id": category_id,
                "name": "person" if category_id == 1 else f"person{category_id}",
                "supercategory": "person",
                "keypoints": [str(k) for k in range(num_keypoints)],
                "skeleton": [],
            }
            for category_id in range(1, num_categories + 1)
        ]
    else:
        categories = [{"id": c, "name": f"c{c}", "supercategory": "s"} for c in range(1, num_categories + 1)]
    cat_ids: list[int] = [c["id"] for c in categories]

    gts: list[dict[str, Any]] = []
    dts: list[dict[str, Any]] = []
    for img in images:
        for _ in range(rng.randint(*annotations_per_image)):
            box = _box(rng, image_width, image_height)
            gt: dict[str, Any] = {
                "id": len(gts) + 1,
                "image_id": img["id"],
                "category_id": rng.choice(cat_ids),
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": int(rng.random() < 0.05),
                "segmentation": [_polygon(*box, rng)],
            }
            if iou_type == "keypoints":
                gt["keypoints"], gt["num_keypoints"] = _keypoints(box, rng, 0.7, num_keypoints)
            gts.append(gt)
            if rng.random() < 0.8:
                dbox = _jitter(box, rng, image_width, image_height)
                dts.append(_detection(img["id"], gt["category_id"], dbox, iou_type, rng, num_keypoints))
        for _ in range(rng.randint(*unmatched_detections_per_image)):
            dts.append(
                _detection(
                    img["id"],
                    rng.choice(cat_ids),
                    _box(rng, image_width, image_height),
                    iou_type,
                    rng,
                    num_keypoints,
                )
            )
    return {"images": images, "annotations": gts, "categories": categories}, dts
