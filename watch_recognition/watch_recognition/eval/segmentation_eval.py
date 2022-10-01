import time
from pathlib import Path

import click
from matplotlib import pyplot as plt
from more_itertools import flatten
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from watch_recognition.label_studio_adapters import (
    load_label_studio_polygon_detection_dataset,
)
from watch_recognition.predictors import (
    HandPredictorLocal,
    KPHeatmapPredictorV2Local,
    RetinanetDetector,
)


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
                split=split,
            )
        ):
            save_file = Path(f"example_predictions/segmentation/{split}_{i}.jpg")
            save_file.parent.mkdir(exist_ok=True)
            plt.figure()
            plt.tight_layout()
            model.predict_mask_and_draw(image_np)
            plt.axis("off")
            plt.savefig(save_file, bbox_inches="tight")

        # TODO any segmentation metrics?
    elapsed = time.perf_counter() - t0
    print(f"Segmentation eval done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
