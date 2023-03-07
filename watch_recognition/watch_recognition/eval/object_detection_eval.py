import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Union

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from watch_recognition.label_studio_adapters import (
    load_label_studio_bbox_detection_dataset,
    load_label_studio_bbox_detection_dataset_with_images,
)
from watch_recognition.predictors import RetinaNetDetector, RetinaNetDetectorLocal
from watch_recognition.train.utils import label_studio_bbox_detection_dataset_to_coco
from watch_recognition.utilities import BBox, iou_bbox_matching

plt.rcParams["font.family"] = "Roboto"


@click.command()
def main():
    t0 = time.perf_counter()
    dataset_path = Path("datasets/watch-faces-local.json")
    label_to_cls = {"WatchFace": 1}
    cls_to_label = {v: k for k, v in label_to_cls.items()}
    detector = RetinaNetDetectorLocal(
        Path("exported_models/detector/serving"), class_to_label_name=cls_to_label
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
        detector_metrics_dir = Path(f"metrics/detector/{split}")
        detector_metrics_dir.mkdir(parents=True, exist_ok=True)

        save_example_predictions(dataset_path, detector, label_to_cls, split)

        calculate_and_save_detection_metrics(
            dataset_path,
            detector,
            label_to_cls,
            split=split,
            save_dir=detector_metrics_dir,
            save_per_image_metrics=False,
            plot=False,
        )

        calculate_and_save_coco_metrics(
            dataset_path,
            detector,
            label_to_cls,
            selected_coco_metrics,
            split,
            save_dir=detector_metrics_dir,
        )
    elapsed = time.perf_counter() - t0
    print(f"Object detection evaluation done in {elapsed:.2f}s")


def save_example_predictions(dataset_path, detector, label_to_cls, split):
    for i, (image, bbox, cls) in enumerate(
        load_label_studio_bbox_detection_dataset_with_images(
            dataset_path,
            label_mapping=label_to_cls,
            max_num_images=3,
            split=split,
            skip_images_without_annotations=False,
        )
    ):
        save_file = Path(f"example_predictions/detector/{split}_{i}.jpg")
        save_file.parent.mkdir(exist_ok=True)
        plt.figure()
        plt.tight_layout()
        detector.predict_and_plot(image)
        plt.axis("off")
        plt.savefig(save_file, bbox_inches="tight")


def calculate_and_save_detection_metrics(
    dataset_path: Union[str, Path],
    detector: RetinaNetDetectorLocal,
    label_to_cls: Dict[str, int],
    split: str,
    save_dir: Path,
    plot: bool = False,
    save_per_image_metrics: bool = False,
):
    all_results = []
    cls_to_label = {v: k for k, v in label_to_cls.items()}
    dateset_gen = load_label_studio_bbox_detection_dataset(
        dataset_path,
        label_mapping=label_to_cls,
        split=split,
        skip_images_without_annotations=False,
    )
    dataset = list(dateset_gen)
    # TODO batched dataset gen an predictions?
    for (
        task_id,
        image_path,
        bboxes,
        cls,
    ) in tqdm(dataset):
        target_bboxes = [
            BBox(*bbox, name=cls_to_label[c]) for bbox, c in zip(bboxes, cls.flatten())
        ]
        with Image.open(image_path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            predicted_bboxes = detector.predict(image)

        target_bboxes = [
            bbox.scale(image.width, image.height) for bbox in target_bboxes
        ]
        matching_dict, unmatched_boxes = iou_bbox_matching(
            target_bboxes,
            predicted_bboxes,
            threshold=0.5,
        )
        if plot:
            plt.figure(figsize=(15, 15))
            plt.tight_layout()
            plt.imshow(image)
            plot_save_dir = Path(f"renders/{split}/")
            plot_save_dir.mkdir(exist_ok=True, parents=True)
            for false_positive in unmatched_boxes:
                false_positive.plot(
                    color="red",
                    linestyle="--",
                    draw_score=True,
                    draw_label=False,
                    linewidth=3,
                )
            for target, prediction in matching_dict.items():
                if prediction is not None:
                    # true positive
                    target.plot(color="green", linewidth=3)
                    prediction.plot(
                        color="green",
                        linestyle="--",
                        draw_score=True,
                        draw_label=False,
                        linewidth=3,
                    )
                else:
                    # false negative
                    plt.axis("off")
                    target.plot(color="orange", linewidth=3)

            plt.savefig(plot_save_dir / f"{task_id}.jpg")
            plt.close()

        n_tp = sum(1 for v in matching_dict.values() if v is not None)
        n_fn = sum(1 for v in matching_dict.values() if v is None)
        n_fp = len(unmatched_boxes)

        precision = n_tp / (n_tp + n_fp) if n_tp else 0
        recall = n_tp / (n_tp + n_fn) if n_tp else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if n_tp else 0
        results = {
            "Task ID": task_id,
            "TP": n_tp,
            "FN": n_fn,
            "FP": n_fp,
            "Precision": precision,
            "Recall": recall,
            "F1": f1_score,
        }
        all_results.append(results)
    df = pd.DataFrame(all_results)
    if save_per_image_metrics:
        df.to_csv(save_dir / "detection_per_image.csv", index=False)
    precision = df["TP"].sum() / (df["TP"].sum() + df["FP"].sum())
    recall = df["TP"].sum() / (df["TP"].sum() + df["FN"].sum())
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Split: {split}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print("-" * 80)
    with open(save_dir / "detection.json", "w") as f:
        json.dump(
            {
                "Precision": float(precision),
                "Recall": float(recall),
                "F1": float(f1),
                "TP": int(df["TP"].sum()),
                "FN": int(df["FN"].sum()),
                "FP": int(df["FP"].sum()),
            },
            f,
            indent=2,
        )


def calculate_and_save_coco_metrics(
    dataset_path: Path,
    detector: RetinaNetDetectorLocal,
    label_to_cls: Dict[str, int],
    selected_coco_metrics: Dict[int, str],
    split: str,
    save_dir: Path,
):
    """Calculate and save COCO metrics for watch face detector"""
    cls_to_label = {v: k for k, v in label_to_cls.items()}
    with tempfile.TemporaryDirectory() as tmp:
        coco_tmp_dataset_file = Path(tmp) / f"coco-kp-{split}.json"
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
    with open(save_dir / "coco.json", "w") as f:
        json.dump(metrics, f, indent=2)
    precision = coco_eval.eval["precision"]
    all_recalls = slice(None, None)
    iou_th_50_idx = int(np.argwhere(coco_eval.params.iouThrs == 0.50))
    iou_th_75_idx = int(np.argwhere(coco_eval.params.iouThrs == 0.75))
    iou_th_95_idx = int(np.argwhere(coco_eval.params.iouThrs == 0.95))
    class_0_idx = 0
    areas_all_idx = 0
    max_dets_100_idx = 2
    pr_50 = precision[
        iou_th_50_idx, all_recalls, class_0_idx, areas_all_idx, max_dets_100_idx
    ]
    pr_75 = precision[
        iou_th_75_idx, all_recalls, class_0_idx, areas_all_idx, max_dets_100_idx
    ]
    pr_95 = precision[
        iou_th_95_idx, all_recalls, class_0_idx, areas_all_idx, max_dets_100_idx
    ]
    df_50 = pd.DataFrame({"Recall": coco_eval.params.recThrs, "Precision": pr_50})

    df_50.to_csv(save_dir / "PR-IoU@0.50.tsv", sep="\t", index=False)
    df_75 = pd.DataFrame({"Recall": coco_eval.params.recThrs, "Precision": pr_75})
    df_75.to_csv(save_dir / "PR-IoU@0.75.tsv", sep="\t", index=False)
    df_95 = pd.DataFrame({"Recall": coco_eval.params.recThrs, "Precision": pr_95})
    df_95.to_csv(save_dir / "PR-IoU@0.95.tsv", sep="\t", index=False)


def generate_coco_annotations_from_model(
    detector: RetinaNetDetector,
    coco_ds_file: Union[str, Path],
    cls_to_label: Dict[int, str],
) -> List[Dict]:
    """Generate coco annotations from a model and dataset file"""
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


if __name__ == "__main__":
    main()
