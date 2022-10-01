import json
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from more_itertools import flatten
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from watch_recognition.label_studio_adapters import (
    load_label_studio_kp_detection_dataset,
)
from watch_recognition.predictors import (
    KPHeatmapPredictorV2Local,
    RetinanetDetector,
    RetinanetDetectorLocal,
)
from watch_recognition.train.utils import label_studio_bbox_detection_dataset_to_coco


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
@click.option("--kp-confidence-threshold", default=0.5, type=float)
def main(kp_confidence_threshold):
    t0 = time.perf_counter()
    detector = RetinanetDetectorLocal(
        Path("models/detector/"), class_to_label_name={0: "WatchFace"}
    )
    # TODO this should be in params.yaml
    label_to_cls = {
        "Top": 0,
        "Center": 1,
        "Crown": 2,
    }
    dataset_path = Path("datasets/watch-faces-local.json")
    cls_to_label = {v: k for k, v in label_to_cls.items()}
    kp_detector = KPHeatmapPredictorV2Local(
        Path("models/keypoint/"),
        class_to_label_name=cls_to_label,
        confidence_threshold=kp_confidence_threshold,
    )
    bbox_labels = ["WatchFace"]

    selected_coco_metrics = {
        0: "AP @IoU=0.50:0.95",
        1: "AP @IoU=0.50",
        2: "AP @IoU=0.75",
        5: "AR @IoU=0.50:0.95",
        6: "AR @IoU=0.50",
        7: "AR @IoU=0.75",
    }
    for split in ["train", "val"]:
        print(f"evaluating {split}")

        for i, (image, kps) in enumerate(
            load_label_studio_kp_detection_dataset(
                dataset_path,
                crop_size=(256, 256),
                bbox_labels=bbox_labels,
                label_mapping=label_to_cls,
                max_num_images=5,
                split=split,
            )
        ):
            save_file = Path(f"example_predictions/keypoint/{split}_{i}.jpg")
            save_file.parent.mkdir(exist_ok=True)
            plt.figure()
            plt.tight_layout()
            kp_detector.predict_and_plot(image)
            plt.axis("off")
            plt.savefig(save_file, bbox_inches="tight")

        coco_tmp_dataset_file = Path(f"/tmp/coco-kp-{split}.json")
        label_studio_bbox_detection_dataset_to_coco(
            dataset_path,
            output_file=coco_tmp_dataset_file,
            label_mapping=label_to_cls,
            split=split,
        )
        results = generate_kp_coco_annotations_from_model(
            detector=detector,
            kp_model=kp_detector,
            coco_ds_file=coco_tmp_dataset_file,
        )
        coco_gt = COCO(coco_tmp_dataset_file)
        metrics = {"Num Images": len(coco_gt.imgs)}

        coco_dt = coco_gt.loadRes(results)
        #
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints")
        # TODO understand what sigmas are appropriate for various watch keypoints
        coco_eval.params.kpt_oks_sigmas = np.array([0.5, 0.5, 0.5]) / 10.0
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        for k, v in selected_coco_metrics.items():
            metrics[v] = coco_eval.stats[k]
        with open(f"metrics/keypoint/coco_{split}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # TODO extract per class KP detection PR curves from coco eval data
        # precision = coco_eval.eval["precision"]
        # all_recalls = slice(None, None)
        # iou_th_50 = int(np.argwhere(coco_eval.params.iouThrs == 0.50))
        # iou_th_75 = int(np.argwhere(coco_eval.params.iouThrs == 0.75))
        # iou_th_95 = int(np.argwhere(coco_eval.params.iouThrs == 0.95))
        # areas_all = 0
        # max_dets_100 = 2
        # pr_50 = precision[iou_th_50, all_recalls, cls, areas_all, max_dets_100]
        # pr_75 = precision[iou_th_75, all_recalls, cls, areas_all, max_dets_100]
        # pr_95 = precision[iou_th_95, all_recalls, cls, areas_all, max_dets_100]
        #
        # df_50 = pd.DataFrame({"Recall": coco_eval.params.recThrs, "Precision": pr_50})
        # df_50.to_csv(f"metrics/keypoint/PR-IoU@0.50_{split}.tsv", sep="\t", index=False)
        #
        # df_75 = pd.DataFrame({"Recall": coco_eval.params.recThrs, "Precision": pr_75})
        # df_75.to_csv(f"metrics/keypoint/PR-IoU@0.75_{split}.tsv", sep="\t", index=False)
        #
        # df_95 = pd.DataFrame({"Recall": coco_eval.params.recThrs, "Precision": pr_95})
        # df_95.to_csv(f"metrics/keypoint/PR-IoU@0.95_{split}.tsv", sep="\t", index=False)
    elapsed = time.perf_counter() - t0
    print(f"Keypoint detecto eval done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
