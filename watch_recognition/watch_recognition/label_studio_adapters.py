import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from watch_recognition.data_preprocessing import load_image
from watch_recognition.utilities import BBox, Point, Polygon, match_objects_to_bboxes


def load_label_studio_polygon_detection_dataset(
    source: Path,
    bbox_labels: List[str],
    label_mapping: Dict[str, int],
    crop_size: Optional[Tuple[int, int]] = (96, 96),
    max_num_images: Optional[int] = None,
    split: Optional[str] = "train",
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with source.open("r") as f:
        tasks = json.load(f)
    if split is not None:
        tasks = [task for task in tasks if task["image"].startswith(split)]
    image_count = 0
    for task in tasks:
        image_bboxes: List[BBox] = []
        polygons: List[Polygon] = []
        image_path = source.parent / task["image"]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if "bbox" in task:
                for obj in task["bbox"]:
                    bbox_label_name = obj["rectanglelabels"][0]
                    if bbox_label_name in bbox_labels:
                        bbox = BBox.from_label_studio_object(obj)
                        image_bboxes.append(
                            bbox,
                        )
            if "polygon" in task:
                for obj in task["polygon"]:
                    poly = Polygon.from_label_studio_object(obj)
                    polygons.append(poly)
            rectangles_to_polygons = match_objects_to_bboxes(image_bboxes, polygons)
            for bbox, matched_polygons in rectangles_to_polygons.items():
                crop_coordinates = bbox.scale(
                    img.width, img.height
                ).convert_to_int_coordinates_tuple("floor")
                crop_img = img.crop(crop_coordinates)
                if crop_size:
                    crop_img = crop_img.resize(crop_size)
                crop_polygons: List[Polygon] = []
                for poly in matched_polygons:
                    crop_polygons.append(
                        poly.translate(-bbox.x_min, -bbox.y_min)
                        .scale(1 / bbox.width, 1 / bbox.height)
                        .scale(crop_img.width, crop_img.height)
                        .rename(label_mapping[poly.name])
                    )

                yield np.array(crop_img), crop_polygons
                image_count += 1
                if image_count == max_num_images:
                    return


def load_label_studio_kp_detection_dataset(
    source: Path,
    bbox_labels: List[str],
    label_mapping: Dict[str, int],
    crop_size: Optional[Tuple[int, int]] = (96, 96),
    max_num_images: Optional[int] = None,
    split: Optional[str] = "train",
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with source.open("r") as f:
        tasks = json.load(f)
    if split is not None:
        tasks = [task for task in tasks if task["image"].startswith(split)]
    image_count = 0
    for task in tasks:
        image_bboxes: List[BBox] = []
        keypoints: List[Point] = []
        image_path = source.parent / task["image"]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if "bbox" in task:
                for obj in task["bbox"]:
                    bbox_label_name = obj["rectanglelabels"][0]
                    if bbox_label_name in bbox_labels:
                        bbox = BBox.from_label_studio_object(obj)

                        image_bboxes.append(
                            bbox,
                        )
            if "kp" in task:
                for obj in task["kp"]:
                    kp = Point.from_label_studio_object(obj)

                    keypoints.append(kp)
            rectangles_to_kps = match_objects_to_bboxes(image_bboxes, keypoints)
            for bbox, kps in rectangles_to_kps.items():
                crop_coordinates = bbox.scale(
                    img.width, img.height
                ).convert_to_int_coordinates_tuple("floor")
                crop_img = img.crop(crop_coordinates)
                if crop_size:
                    crop_img = crop_img.resize(crop_size)
                crop_kps: List[Point] = []
                for kp in kps:
                    crop_kps.append(
                        kp.translate(-bbox.x_min, -bbox.y_min)
                        .scale(1 / bbox.width, 1 / bbox.height)
                        .scale(crop_img.width, crop_img.height)
                    )
                kp_array = np.array(
                    [
                        (*kp.as_coordinates_tuple, label_mapping[kp.name])
                        for kp in crop_kps
                    ]
                )

                yield np.array(crop_img), kp_array
                image_count += 1
                if image_count == max_num_images:
                    return


def load_label_studio_bbox_detection_dataset(
    source: Path,
    label_mapping: Optional[Dict[str, int]],
    image_size: Optional[Tuple[int, int]] = (400, 400),
    max_num_images: Optional[int] = None,
    split: Optional[str] = "train",
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    dataset_gen = _load_label_studio_bbox_detection_dataset(
        source=source,
        label_mapping=label_mapping,
        max_num_images=max_num_images,
        split=split,
    )
    for (image_path, image_bboxes, class_labels) in dataset_gen:
        image_np = load_image(str(image_path), image_size=image_size).astype("uint8")
        yield image_np, image_bboxes, class_labels


# TODO rename and make public
def _load_label_studio_bbox_detection_dataset(
    source: Path,
    label_mapping: Optional[Dict[str, int]],
    max_num_images: Optional[int] = None,
    split: Optional[str] = "train",
    skip_empty: bool = True,
) -> Iterator[Tuple[Path, np.ndarray, np.ndarray]]:
    with source.open("r") as f:
        tasks = json.load(f)
    if split is not None:
        tasks = [task for task in tasks if task["image"].startswith(split)]

    if max_num_images:
        tasks = tasks[:max_num_images]

    for task in tasks:
        if "bbox" not in task and skip_empty:
            continue
        class_labels, image_bboxes = _extract_bboxes_from_task(label_mapping, task)
        image_path = source.parent / task["image"]
        image_bboxes = np.array(image_bboxes).reshape(-1, 4).astype("float32")
        class_labels = np.array(class_labels).reshape(-1, 1).astype("int32")
        yield image_path, image_bboxes, class_labels


def _extract_bboxes_from_task(label_mapping, task):
    image_bboxes = []
    class_labels = []
    if "bbox" not in task:
        return np.empty(shape=(0, 4)), np.empty(shape=(0, 1))
    for obj in task["bbox"]:
        # label studio keeps
        bbox = BBox.from_ltwh(
            obj["x"],
            obj["y"],
            obj["width"],
            obj["height"],
            obj["rectanglelabels"][0],
        ).scale(1 / 100, 1 / 100)

        image_bboxes.append([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
        class_labels.append(label_mapping[bbox.name])
    return class_labels, image_bboxes
