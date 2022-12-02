import os
from concurrent import futures
from concurrent.futures import as_completed
from pathlib import Path
from shutil import copy, rmtree
from typing import Any, Dict

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from watch_recognition.predictors import HandPredictorGRPC, RetinaNetDetectorGRPC

host = "localhost:8500"
detector = RetinaNetDetectorGRPC(
    host=host,
    model_name="detector",
    class_to_label_name={0: "WatchFace"},
)
hand_predictor = HandPredictorGRPC(
    host=host, model_name="segmentation", confidence_threshold=0.05
)


def score_image(img_path) -> Dict[str, Any]:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        bboxes = detector.predict(img)
        if not bboxes:
            return {
                "img_path": img_path,
                "polygon_score": -1.0,
                "bbox_score": -1.0,
            }
        polygon_scores = []
        bbox_scores = []
        for box in bboxes:
            bbox_scores.append(box.score)
            with img.crop(box=box.as_coordinates_tuple) as crop:
                polygon = hand_predictor.predict(crop)
                polygon_scores.append(polygon.score)
        return {
            "img_path": img_path,
            "polygon_score": float(np.mean(polygon_scores)),
            "bbox_score": float(np.mean(bbox_scores)),
        }


@click.command()
@click.option("--data-dir", type=click.Path(exists=True))
@click.option("--run-concurrently", is_flag=True)
@click.option("--output-path", type=click.Path())
@click.option("--num-samples", type=int)
@click.option("--clean-save-dir", is_flag=True)
def main(data_dir, run_concurrently, output_path, num_samples, clean_save_dir):
    source_dir = Path(data_dir)
    output_path = Path(output_path) / "train"
    output_path.mkdir(parents=True, exist_ok=True)
    if clean_save_dir:
        rmtree(output_path)

    image_paths = list(source_dir.rglob("*.jp*g"))
    records = []
    if run_concurrently:
        with futures.ThreadPoolExecutor(os.cpu_count() // 2 or 1) as executor:
            pass
            task_futures = []
            try:
                for img_path in image_paths:
                    future = executor.submit(score_image, img_path)
                    task_futures.append(future)
                for future in tqdm(as_completed(task_futures), total=len(task_futures)):
                    records.append(future.result())
            except KeyboardInterrupt:
                print("cancelling futures")
                for future in task_futures:
                    future.cancel()
                for future in task_futures:
                    if not future.done():
                        print(f"waiting for {future} to complete...")
                raise
    else:
        for img_path in tqdm(image_paths):
            record = score_image(img_path)

            records.append(record)
    df = pd.DataFrame(records)

    df = df[df["bbox_score"] > 0]
    df = df.sort_values("polygon_score", ascending=True)
    print(df.head(10))

    print(f"Saving {num_samples} worst images for bbox predictions to {output_path}")
    for i, row in (
        df.sort_values("bbox_score", ascending=True).head(num_samples).iterrows()
    ):
        img_path = row["img_path"]
        copy(img_path, output_path / img_path.name)

    print(
        f"Saving {num_samples*2} worst images for polygon predictions to {output_path}"
    )
    for i, row in (
        df.sort_values("polygon_score", ascending=True).head(num_samples * 2).iterrows()
    ):
        img_path = row["img_path"]
        copy(img_path, output_path / img_path.name)


if __name__ == "__main__":
    main()
