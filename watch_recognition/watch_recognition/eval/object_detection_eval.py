import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tensorflow import keras
from tqdm import tqdm

from watch_recognition.train.object_detection_task import (
    label_studio_dataset_to_coco,
    resize_and_pad_image,
)
from watch_recognition.utilities import BBox


def generate_coco_annotations_from_model(
    model, coco_ds_file, output_file, cls_to_label
):
    coco = COCO(coco_ds_file)
    annotations = []
    object_counter = 1
    for image_id in tqdm(coco.imgs):
        with Image.open(coco.imgs[image_id]["coco_url"]) as img:
            img_arr = np.array(img)
            input_image, ratio = prepare_image(img_arr)
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
            predictions = []
            for box, cls, score in zip(nmsed_boxes, nmsed_classes, nmsed_scores):
                predictions.append(BBox(*box, name=cls_to_label[int(cls)], score=score))

            coco_predictions = []
            for box in predictions:
                coco_predictions.append(
                    box.to_coco_object(
                        image_id=image_id,
                        object_id=object_counter,
                        category_id=coco.cats[cls]["id"],
                    )
                )
                object_counter += 1
            annotations.extend(coco_predictions)
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=2, cls=NpEncoder)


class NpEncoder(json.JSONEncoder):
    """Source: https://bobbyhadz.com/blog/python-typeerror-object-of-type-float32-is-not-json-serializable"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None, max_side=384)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


def main():
    model = keras.models.load_model("models/detector/")
    model.summary()

    dataset_path = Path("datasets/watch-faces-local.json")
    label_to_cls = {"WatchFace": 1}
    # model is trained with 0 as a valid cls but coco metrics ignore cls 0
    cls_to_label = {0: "WatchFace"}

    selected_coco_metrics = {
        0: "mAP @IoU=0.50:0.95",
        1: "AP @IoU=0.50",
        2: "AP @IoU=0.75",
        6: "AR @maxDets=1",
        7: "AR @maxDets=10",
        8: "AR @maxDets=100",
    }
    for split in ["train"]:
        print(f"evaluating {split}")
        coco_tmp_dataset_file = Path(f"/tmp/coco-{split}.json")
        coco_tmp_detections_file = Path(f"/tmp/coco-{split}-results.json")
        label_studio_dataset_to_coco(
            dataset_path,
            output_file=coco_tmp_dataset_file,
            label_mapping=label_to_cls,
            split="train",
        )
        generate_coco_annotations_from_model(
            model,
            coco_tmp_dataset_file,
            coco_tmp_detections_file,
            cls_to_label=cls_to_label,
        )

        coco_gt = COCO(coco_tmp_dataset_file)
        coco_dt = coco_gt.loadRes(coco_tmp_detections_file)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {"Num Images": len(coco_gt.imgs)}

        for k, v in selected_coco_metrics.items():
            metrics[v] = coco_eval.stats[k]
        with open(f"metrics/coco_{split}.json", "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
