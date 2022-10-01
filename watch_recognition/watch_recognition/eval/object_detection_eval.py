import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from watch_recognition.label_studio_adapters import (
    load_label_studio_bbox_detection_dataset,
)
from watch_recognition.predictors import RetinanetDetector, RetinanetDetectorLocal
from watch_recognition.train.utils import label_studio_bbox_detection_dataset_to_coco


def generate_coco_annotations_from_model(
    detector: RetinanetDetector, coco_ds_file, cls_to_label
):
    coco = COCO(coco_ds_file)
    annotations = []
    object_counter = 1
    label_to_cls = {v: k for k, v in cls_to_label.items()}
    for image_id in tqdm(coco.imgs):
        with Image.open(coco.imgs[image_id]["coco_url"]) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
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
    t0 = time.perf_counter()
    dataset_path = Path("datasets/watch-faces-local.json")
    label_to_cls = {"WatchFace": 1}
    # model is trained with 0 as a valid cls but coco metrics ignore cls 0
    cls_to_label = {0: "WatchFace"}
    detector = RetinanetDetectorLocal(
        Path("models/detector/"), class_to_label_name=cls_to_label
    )

    selected_coco_metrics = {
        0: "AP @IoU=0.50:0.95",
        1: "AP @IoU=0.50",
        2: "AP @IoU=0.75",
        3: "AP @IoU=0.95",
        6: "AR @maxDets=1",
        7: "AR @maxDets=10",
        8: "AR @maxDets=100",
    }
    for split in ["train", "val"]:
        print(f"evaluating {split}")

        for i, (image, bbox, cls) in enumerate(
            load_label_studio_bbox_detection_dataset(
                dataset_path,
                label_mapping=label_to_cls,
                max_num_images=5,
                split=split,
            )
        ):
            save_file = Path(f"example_predictions/detector/{split}_{i}.jpg")
            save_file.parent.mkdir(exist_ok=True)
            plt.figure()
            plt.tight_layout()
            detector.predict_and_plot(image)
            plt.axis("off")
            plt.savefig(save_file, bbox_inches="tight")

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

        coco_dt = coco_gt.loadRes(results)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        for k, v in selected_coco_metrics.items():
            metrics[v] = coco_eval.stats[k]
        with open(f"metrics/detector/coco_{split}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        precision = coco_eval.eval["precision"]
        all_recalls = slice(None, None)
        iou_th_50 = int(np.argwhere(coco_eval.params.iouThrs == 0.50))
        iou_th_75 = int(np.argwhere(coco_eval.params.iouThrs == 0.75))
        iou_th_95 = int(np.argwhere(coco_eval.params.iouThrs == 0.95))
        class_0 = 0
        areas_all = 0
        max_dets_100 = 2
        pr_50 = precision[iou_th_50, all_recalls, class_0, areas_all, max_dets_100]
        pr_75 = precision[iou_th_75, all_recalls, class_0, areas_all, max_dets_100]
        pr_95 = precision[iou_th_95, all_recalls, class_0, areas_all, max_dets_100]

        df_50 = pd.DataFrame({"Recall": coco_eval.params.recThrs, "Precision": pr_50})
        df_50.to_csv(f"metrics/detector/PR-IoU@0.50_{split}.tsv", sep="\t", index=False)

        df_75 = pd.DataFrame({"Recall": coco_eval.params.recThrs, "Precision": pr_75})
        df_75.to_csv(f"metrics/detector/PR-IoU@0.75_{split}.tsv", sep="\t", index=False)

        df_95 = pd.DataFrame({"Recall": coco_eval.params.recThrs, "Precision": pr_95})
        df_95.to_csv(f"metrics/detector/PR-IoU@0.95_{split}.tsv", sep="\t", index=False)
    elapsed = time.perf_counter() - t0
    print(f"Object detection evaluation done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
