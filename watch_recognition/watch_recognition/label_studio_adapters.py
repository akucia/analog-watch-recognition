import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from watch_recognition.utilities import BBox, Point, match_bboxes_to_points


def load_label_studio_kp_detection_dataset(
    source: Path,
    bbox_labels: List[str],
    label_mapping: Optional[Dict[str, int]],
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
            for obj in task["bbox"]:
                bbox_label_name = obj["rectanglelabels"][0]
                if bbox_label_name in bbox_labels:
                    bbox = BBox.from_label_studio_object(obj)

                    image_bboxes.append(
                        bbox,
                    )
            for obj in task["kp"]:
                kp = Point.from_label_studio_object(obj)

                keypoints.append(kp)
            rectangles_to_kps = match_bboxes_to_points(image_bboxes, keypoints)
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
                crops_array = np.array(
                    [
                        (*kp.as_coordinates_tuple, label_mapping[kp.name])
                        for kp in crop_kps
                    ]
                )

                yield np.array(crop_img), crops_array
                image_count += 1
                if image_count == max_num_images:
                    return
