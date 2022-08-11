import json
from pathlib import Path

import tensorflow as tf
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tensorflow import keras
from tqdm import tqdm

from watch_recognition.predictors import RetinanetDetector
from watch_recognition.train.object_detection_task import (
    load_label_studio_bbox_detection_dataset,
    visualize_detections,
)
from watch_recognition.train.utils import label_studio_bbox_detection_dataset_to_coco
from watch_recognition.utilities import retinanet_prepare_image


def generate_coco_annotations_from_model(
    detector: RetinanetDetector, coco_ds_file, cls_to_label
):
    coco = COCO(coco_ds_file)
    annotations = []
    object_counter = 1
    label_to_cls = {v: k for k, v in cls_to_label.items()}
    for image_id in tqdm(coco.imgs):
        with Image.open(coco.imgs[image_id]["coco_url"]) as img:
            predictions = detector.predict(img)

            coco_predictions = []
            for box in predictions:
                coco_predictions.append(
                    box.to_coco_object(
                        image_id=image_id,
                        object_id=object_counter,
                        category_id=coco.cats[label_to_cls[box.name]]["id"],
                    )
                )
                object_counter += 1
            annotations.extend(coco_predictions)
    return annotations


def main():
    model = keras.models.load_model("models/detector/")

    dataset_path = Path("datasets/watch-faces-local.json")
    label_to_cls = {"WatchFace": 1}
    # model is trained with 0 as a valid cls but coco metrics ignore cls 0
    cls_to_label = {0: "WatchFace"}
    detector = RetinanetDetector(
        Path("models/detector/"), class_to_label_name=cls_to_label
    )

    selected_coco_metrics = {
        0: "AP @IoU=0.50:0.95",
        1: "AP @IoU=0.50",
        2: "AP @IoU=0.75",
        6: "AR @maxDets=1",
        7: "AR @maxDets=10",
        8: "AR @maxDets=100",
    }
    for split in ["train", "val"]:
        print(f"evaluating {split}")

        for i, (image, bbox, cls) in enumerate(
            load_label_studio_bbox_detection_dataset(
                dataset_path,
                image_size=(384, 384),
                label_mapping=label_to_cls,
                max_num_images=5,
                split=split,
            )
        ):
            image = tf.cast(image, dtype=tf.float32)
            input_image, ratio = retinanet_prepare_image(image)
            ratio = ratio.numpy()
            nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = model.predict(
                input_image
            )
            nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
                nmsed_boxes[0],
                nmsed_scores[0],
                nmsed_classes[0],
                valid_detections[0],
            )
            nmsed_boxes = nmsed_boxes[:valid_detections]
            nmsed_classes = nmsed_classes[:valid_detections]
            nmsed_scores = nmsed_scores[:valid_detections]
            nmsed_boxes = nmsed_boxes / ratio
            class_names = [cls_to_label[x] for x in nmsed_classes[:valid_detections]]
            save_file = Path(f"example_predictions/{split}_{i}.jpg")
            save_file.parent.mkdir(exist_ok=True)
            visualize_detections(
                image,
                nmsed_boxes[:valid_detections] / ratio,
                class_names,
                nmsed_scores[:valid_detections],
                savefile=save_file,
            )
        coco_tmp_dataset_file = Path(f"/tmp/coco-{split}.json")
        label_studio_bbox_detection_dataset_to_coco(
            dataset_path,
            output_file=coco_tmp_dataset_file,
            label_mapping=label_to_cls,
            split=split,
        )
        results = generate_coco_annotations_from_model(
            detector,
            coco_tmp_dataset_file,
            cls_to_label=cls_to_label,
        )
        coco_gt = COCO(coco_tmp_dataset_file)
        metrics = {"Num Images": len(coco_gt.imgs)}
        if results:

            coco_dt = coco_gt.loadRes(results)

            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            for k, v in selected_coco_metrics.items():
                metrics[v] = coco_eval.stats[k]
        else:
            for k, v in selected_coco_metrics.items():
                metrics[v] = 0
        with open(f"metrics/coco_{split}.json", "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()