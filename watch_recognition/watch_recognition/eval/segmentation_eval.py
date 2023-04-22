import json
import time
from pathlib import Path
from typing import Dict, List, Union

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from rich.console import Console
from rich.table import Table
from sklearn.metrics import classification_report
from tqdm import tqdm

from watch_recognition.label_studio_adapters import (
    load_label_studio_polygon_detection_dataset,
)
from watch_recognition.predictors import HandPredictor, HandPredictorLocal

plt.rcParams["font.family"] = "Roboto"


def calculate_and_save_segmentation_metrics(
    dataset_path: Union[str, Path],
    detector: HandPredictor,
    label_to_cls: Dict[str, int],
    split: str,
    save_dir: Path,
    bbox_labels: List[str],
    plot: bool = False,
    save_per_image_metrics: bool = False,
):
    all_results = []
    all_ious = []
    dataset_gen = load_label_studio_polygon_detection_dataset(
        dataset_path,
        crop_size=None,
        bbox_labels=bbox_labels,
        label_mapping=label_to_cls,
        split=split,
    )
    dataset = list(dataset_gen)
    y_targets = []
    y_preds = []

    for example_id, image_array, target_polygons in tqdm(dataset):
        with Image.fromarray(image_array) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            predicted_polygon = detector.predict(image)
        predicted_mask = predicted_polygon.to_mask(*image.size, 1)
        # for now let's assume there's only one polygon
        target_mask = target_polygons[0].to_mask(*image.size, 1)

        iou_score = iou_score_from_masks(predicted_mask, target_mask)

        y_targets.append(target_mask.flatten().astype(int))
        y_preds.append(predicted_mask.flatten().astype(int))
        all_ious.append(iou_score)

        if save_per_image_metrics:
            all_results.append(
                {
                    "Task ID": example_id,
                    "iou": iou_score,
                }
            )
        if plot:
            plt.figure(figsize=(15, 15))
            plt.tight_layout()
            plt.imshow(image)
            plot_save_dir = Path(f"renders/hands/{split}/")
            plot_save_dir.mkdir(exist_ok=True, parents=True)
            # TODO color should be based on IOU score or other metric
            for target in target_polygons:
                target.plot(
                    color="red",
                    linestyle="-",
                    # draw_score=True,
                    draw_label=False,
                    linewidth=3,
                    alpha=0.5,
                )

            predicted_polygon.plot(
                color="green",
                linestyle="--",
                # draw_score=True,
                draw_label=False,
                linewidth=3,
                alpha=0.5,
            )
            plt.title(f"IOU: {iou_score:.3f}")
            plt.axis("off")
            plt.savefig(plot_save_dir / f"{example_id}.jpg")
            plt.close()

    df = pd.DataFrame(all_results)
    if save_per_image_metrics:
        df.to_csv(save_dir / "segmentation_eval_per_image.csv", index=False)
    y_targets = np.concatenate(y_targets).reshape(-1, 1)
    y_preds = np.concatenate(y_preds).reshape(-1, 1)

    report = classification_report(
        y_targets, y_preds, target_names=["background", "hands"], output_dict=True
    )
    # print sklearn classification report as rich table
    table = Table(title=f"Segmentation metrics {split}")
    table.add_column("class")
    table.add_column("precision")
    table.add_column("recall")
    table.add_column("f1-score")
    table.add_column("support")
    for cls, metrics in report.items():
        if cls == "macro avg":
            table.add_row("", "", "", "", "")
        if cls == "accuracy":
            continue
        table.add_row(
            cls,
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1-score']:.3f}",
            f"{metrics['support']:.0f}",
        )

    console = Console()
    console.print(table)
    report["mIOU"] = np.mean(all_ious)
    report["Num Images"] = len(dataset)
    with open(save_dir / "segmentation.json", "w") as f:
        json.dump(report, f, indent=2)


def iou_score_from_masks(predicted_mask: np.ndarray, target_mask: np.ndarray) -> float:
    intersection = np.logical_and(target_mask, predicted_mask).sum()
    union = np.logical_or(target_mask, predicted_mask).sum()
    if intersection == 0:
        return 0.0
    iou_score = intersection / union
    return iou_score


@click.command()
@click.option("--confidence-threshold", default=0.5, type=float)
@click.option("--save-plots", is_flag=True)
@click.option("--save-per-image-metrics", is_flag=True)
def main(confidence_threshold: float, save_plots: bool, save_per_image_metrics: bool):
    t0 = time.perf_counter()
    # TODO this should be in params.yaml
    label_to_cls = {
        "Hands": 1,
    }
    dataset_path = Path("datasets/watch-faces-local.json")
    model = HandPredictorLocal(
        Path("exported_models/segmentation/hands/default"),
        confidence_threshold=confidence_threshold,
    )

    bbox_labels = ["WatchFace"]
    for split in ["train", "val"]:
        print(f"evaluating {split}")
        metrics_dir = Path(f"metrics/segmentation/hands/{split}")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        save_example_predictions(bbox_labels, dataset_path, label_to_cls, model, split)

        calculate_and_save_segmentation_metrics(
            dataset_path,
            model,
            label_to_cls,
            split=split,
            save_dir=metrics_dir,
            save_per_image_metrics=save_per_image_metrics,
            plot=save_plots,
            bbox_labels=bbox_labels,
        )

    elapsed = time.perf_counter() - t0
    print(f"Segmentation eval done in {elapsed:.2f}s")


def save_example_predictions(bbox_labels, dataset_path, label_to_cls, model, split):
    examples_path = Path("example_predictions/segmentation/hands/")
    examples_path.mkdir(exist_ok=True, parents=True)
    for i, (example_id, image_np, polygons) in enumerate(
        load_label_studio_polygon_detection_dataset(
            dataset_path,
            crop_size=None,
            bbox_labels=bbox_labels,
            label_mapping=label_to_cls,
            max_num_images=5,
            split=split,
        )
    ):
        save_file = Path(examples_path / f"{split}_{i}.jpg")
        save_file.parent.mkdir(exist_ok=True)
        plt.figure()
        plt.tight_layout()
        model.predict_mask_and_draw(image_np)
        plt.axis("off")
        plt.savefig(save_file, bbox_inches="tight")


if __name__ == "__main__":
    main()
