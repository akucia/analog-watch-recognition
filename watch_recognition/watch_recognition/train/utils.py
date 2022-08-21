import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from more_itertools import flatten
from tqdm import tqdm

from watch_recognition.data_preprocessing import load_image
from watch_recognition.utilities import BBox, Point, match_objects_to_bboxes


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
        bboxes = []
        keypoints = []
        # TODO this is repeated in some other place too
        if "bbox" in task:
            for i, obj in enumerate(task["bbox"]):
                # label studio keeps
                label_name = obj["rectanglelabels"][0]
                if label_name not in categories:
                    categories[label_name] = {
                        "supercategory": "watch",
                        "id": len(categories),
                        "name": label_name,
                        "keypoints": [],
                    }
            bbox = BBox.from_label_studio_object(obj).scale(img_width, img_height)
            bboxes.append(bbox)
        if "kp" in task:
            for i, obj in enumerate(task["kp"]):
                kp = Point.from_label_studio_object(obj).scale(img_width, img_height)
                keypoints.append(kp)

        bboxes_to_kps = match_objects_to_bboxes(bboxes, keypoints)
        # It's a mess, but it works
        # Categories could be a separate class with appropriate API and
        # serialization
        for bbox, kps in bboxes_to_kps.items():
            category = categories[bbox.name]
            for kp in kps:
                if kp.name not in category["keypoints"]:
                    category["keypoints"].append(kp.name)
            kp_to_name = {kp.name: [kp.x, kp.y, 2] for kp in kps}
            object_keypoints = []
            for kp_name in category["keypoints"]:
                object_keypoints.append(kp_to_name.get(kp_name, [0, 0, 0]))
            object_keypoints = list(flatten(object_keypoints))
            coco_object = bbox.rename(category["id"]).to_coco_object(
                image_id=image_id, object_id=object_counter
            )
            coco_object["num_keypoints"] = len(object_keypoints) // 3
            coco_object["keypoints"] = object_keypoints

            annotations.append(coco_object)
            object_counter += 1
    # 2nd pass on the objects might be required to pad with any missing keypoints
    cat_id_to_category = {cat["id"]: cat for cat in categories.values()}
    for obj in annotations:
        category = cat_id_to_category[obj["category_id"]]
        n_expected_keypoints = len(category["keypoints"])
        missing_keypoints = n_expected_keypoints - obj["num_keypoints"]
        for _ in range(missing_keypoints):
            obj["keypoints"].extend([0, 0, 0])
            obj["num_keypoints"] += 1
    dataset = {
        "info": info,
        "images": images,
        "categories": list(categories.values()),
        "annotations": annotations,
    }
    with Path(output_file).open("w") as f:
        json.dump(dataset, f, indent=2)
