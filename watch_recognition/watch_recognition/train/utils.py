import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from watch_recognition.data_preprocessing import load_image
from watch_recognition.utilities import BBox


def unison_shuffled_copies(a, b, seed=42):
    """https://stackoverflow.com/a/4602224/8814045"""
    np.random.seed(seed)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def label_studio_bbox_detection_dataset_to_coco(
    source: Path,
    output_file: Union[Path, str],
    label_mapping: Optional[Dict[str, int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    max_num_images: Optional[int] = None,
    split: Optional[str] = "train",
):  # TODO add types
    with source.open("r") as f:
        tasks = json.load(f)
    if split is not None:
        tasks = [task for task in tasks if task["image"].startswith(split)]
    info = {}
    images = []
    categories = {}
    annotations = []
    if max_num_images:
        tasks = tasks[:max_num_images]
    object_counter = 1
    for task in tqdm(tasks):
        image_path = source.parent / task["image"]
        img_np = load_image(
            str(image_path), image_size=image_size, preserve_aspect_ratio=True
        )
        img_height = img_np.shape[0]
        img_width = img_np.shape[1]
        image_id = image_path.stem
        images.append(
            {
                "file_name": image_path.name,
                "coco_url": str(image_path),
                "id": image_id,
                "height": img_height,
                "width": img_width,
            }
        )

        for i, obj in enumerate(task["bbox"]):
            # label studio keeps
            label_name = obj["rectanglelabels"][0]
            if label_name not in categories:
                categories[label_name] = {
                    "supercategory": "watch",
                    "id": len(categories),
                    "name": label_name,
                }
            bbox = (
                BBox.from_ltwh(
                    obj["x"],
                    obj["y"],
                    obj["width"],
                    obj["height"],
                    categories[label_name]["id"],
                )
                .scale(1 / 100, 1 / 100)
                .scale(img_width, img_height)
            )

            annotations.append(
                bbox.to_coco_object(image_id=image_id, object_id=object_counter)
            )
            object_counter += 1

    dataset = {
        "info": info,
        "images": images,
        "categories": list(categories.values()),
        "annotations": annotations,
    }
    with Path(output_file).open("w") as f:
        json.dump(dataset, f, indent=2)
