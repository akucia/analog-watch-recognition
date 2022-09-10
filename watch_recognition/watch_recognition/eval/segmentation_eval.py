import json
import time
from pathlib import Path

import click
import cv2
import numpy as np
from more_itertools import flatten
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from watch_recognition.label_studio_adapters import (
    load_label_studio_kp_detection_dataset,
    load_label_studio_polygon_detection_dataset,
)
from watch_recognition.predictors import (
    HandPredictorLocal,
    KPHeatmapPredictorV2Local,
    RetinanetDetector,
    RetinanetDetectorLocal,
)
from watch_recognition.train.segmentation_task import encode_polygon_to_mask
from watch_recognition.train.utils import label_studio_bbox_detection_dataset_to_coco
from watch_recognition.visualization import visualize_keypoints, visualize_masks


def generate_kp_coco_annotations_from_model(
    detector: RetinanetDetector,
    kp_model: KPHeatmapPredictorV2Local,
    coco_ds_file,
):
    coco = COCO(coco_ds_file)
    category_name_to_cat = {cat["name"]: cat for cat in coco.cats.values()}
    annotations = []
    object_counter = 1
    for image_id in tqdm(coco.imgs):
        with Image.open(coco.imgs[image_id]["coco_url"]) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            predictions = detector.predict(img)

            coco_predictions = []
            for box in predictions:
                points = kp_model.predict_from_image_and_bbox(img, box)
                coco_box = box.to_coco_object(
                    image_id=image_id,
                    object_id=object_counter,
                    category_id=category_name_to_cat[box.name]["id"],
                )
                # TODO possibly convert to integers
                points = {kp.name: [float(kp.x), float(kp.y), 2] for kp in points}
                coco_keypoints = [
                    points.get(kp_name, [0, 0, 0])
                    for kp_name in category_name_to_cat[box.name]["keypoints"]
                ]
                coco_keypoints = list(flatten(coco_keypoints))
                coco_box["num_keypoints"] = len(coco_keypoints) // 3
                coco_box["keypoints"] = coco_keypoints
                coco_predictions.append(coco_box)
                object_counter += 1
            annotations.extend(coco_predictions)
    return annotations


@click.command()
@click.option("--confidence-threshold", default=0.5, type=float)
def main(confidence_threshold):
    t0 = time.perf_counter()
    # TODO this should be in params.yaml
    label_to_cls = {
        "Hands": 0,
    }
    dataset_path = Path("datasets/watch-faces-local.json")
    cls_to_label = {v: k for k, v in label_to_cls.items()}
    model = HandPredictorLocal(
        Path("models/segmentation/"),
        confidence_threshold=confidence_threshold,
    )

    bbox_labels = ["WatchFace"]
    for split in ["train", "val"]:
        print(f"evaluating {split}")

        for i, (image_np, polygons) in enumerate(
            load_label_studio_polygon_detection_dataset(
                dataset_path,
                crop_size=(96, 96),
                bbox_labels=bbox_labels,
                label_mapping=label_to_cls,
                max_num_images=5,
                split="train",
            )
        ):
            results = model.model.predict(np.expand_dims(image_np, axis=0), verbose=0)[
                0
            ]

            save_file = Path(f"example_predictions/segmentation/{split}_{i}.jpg")
            save_file.parent.mkdir(exist_ok=True)
            masks = []
            for cls, name in cls_to_label.items():
                mask = results[:, :, cls] > confidence_threshold
                mask = cv2.resize(
                    mask.astype("uint8"),
                    image_np.shape[:2][::-1],
                    interpolation=cv2.INTER_NEAREST,
                ).astype("bool")
                masks.append(mask)

            visualize_masks(image_np, masks, savefile=save_file)

    elapsed = time.perf_counter() - t0
    print(f"Segmentation eval done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
