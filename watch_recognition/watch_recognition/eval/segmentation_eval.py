import time
from pathlib import Path

import click
from matplotlib import pyplot as plt

from watch_recognition.label_studio_adapters import (
    load_label_studio_polygon_detection_dataset,
)
from watch_recognition.predictors import HandPredictorLocal


@click.command()
@click.option("--confidence-threshold", default=0.5, type=float)
@click.option("--save-plots", is_flag=True)
def main(confidence_threshold: float, save_plots: bool):
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
        if save_plots:
            _save_example_predictions(
                bbox_labels, dataset_path, label_to_cls, model, split
            )

        # TODO any segmentation metrics?
    elapsed = time.perf_counter() - t0
    print(f"Segmentation eval done in {elapsed:.2f}s")


def _save_example_predictions(bbox_labels, dataset_path, label_to_cls, model, split):
    examples_path = Path("example_predictions/segmentation/hands/")
    examples_path.mkdir(exist_ok=True, parents=True)
    for i, (image_np, polygons) in enumerate(
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
