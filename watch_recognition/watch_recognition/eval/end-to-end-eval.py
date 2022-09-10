import json
import os
import time
from concurrent import futures
from concurrent.futures import as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import click
import pandas as pd
from PIL import Image
from tqdm import tqdm

from watch_recognition.predictors import (
    HandPredictorLocal,
    KPHeatmapPredictorV2Local,
    RetinanetDetectorLocal,
    TimePredictor,
)
from watch_recognition.utilities import BBox, iou_bbox_matching

SPLITS = ["train", "val"]


def str_to_hours_and_minutes(time_str: str) -> Tuple[int, int]:
    hours, minutes = time_str.split(":")
    return int(hours), int(minutes)


def total_minutes_diff(t1: str, t2: str) -> int:
    hours_minutes_1 = str_to_hours_and_minutes(t1)
    hours_minutes_2 = str_to_hours_and_minutes(t2)
    datetime_1 = datetime(
        hour=hours_minutes_1[0], minute=hours_minutes_1[1], year=2000, month=1, day=1
    )
    datetime_2 = datetime(
        hour=hours_minutes_2[0], minute=hours_minutes_2[1], year=2000, month=1, day=1
    )
    return int(abs((datetime_1 - datetime_2).total_seconds()) / 60)


@click.command()
@click.option("--concurrent", is_flag=True)
def main(concurrent: bool = False):
    t0 = time.perf_counter()
    time_predictor = TimePredictor(
        detector=RetinanetDetectorLocal(
            Path("models/detector/"),
            class_to_label_name={0: "WatchFace"},
        ),
        kp_predictor=KPHeatmapPredictorV2Local(
            Path("models/keypoint/"),
            class_to_label_name={
                0: "Top",
                1: "Center",
                2: "Crown",
            },
            confidence_threshold=0.5,
        ),
        hand_predictor=HandPredictorLocal(
            Path("models/segmentation/"),
        ),
    )
    source = Path("datasets/watch-faces-local.json")

    records = []
    for split in SPLITS:
        print(f"evaluating {split}")
        image_paths, all_targets = load_bboxes_with_transcription(source, split)
        data = zip(image_paths, all_targets)
        if concurrent:
            with futures.ThreadPoolExecutor(os.cpu_count() // 3 or 1) as executor:
                task_futures = []
                try:
                    for img_path, targets in data:
                        future = executor.submit(
                            _evaluate_on_single_image,
                            img_path,
                            targets,
                            split,
                            time_predictor,
                        )
                        task_futures.append(future)
                    for future in tqdm(
                        as_completed(task_futures), total=len(task_futures)
                    ):
                        records.extend(future.result())
                except KeyboardInterrupt:
                    print("cancelling futures")
                    for future in task_futures:
                        future.cancel()
                    for future in task_futures:
                        if not future.done():
                            print(f"waiting for {future} to complete...")
                    raise
        else:
            for img_path, targets in tqdm(data, total=len(image_paths)):
                evaluation_records = _evaluate_on_single_image(
                    img_path, targets, split, time_predictor
                )
                records.extend(evaluation_records)
    df = pd.DataFrame(records)
    df.to_csv("metrics/end_2_end_eval.csv", index=False)
    summary_metrics = {}
    for split in SPLITS:
        split_df = df[df["split"] == split]
        one_minute_acc = (split_df["total_minutes_diff"] <= 1).sum() / len(split_df)
        ten_minutes_acc = (split_df["total_minutes_diff"] <= 10).sum() / len(split_df)
        sixty_minutes_acc = (split_df["total_minutes_diff"] <= 60).sum() / len(split_df)
        summary_metrics[split] = {
            "1-min_acc": one_minute_acc,
            "10-min_acc": ten_minutes_acc,
            "60-min_acc": sixty_minutes_acc,
        }
    with Path("./metrics/end_2_end_summary.json").open("w") as f:
        json.dump(summary_metrics, f, indent=2)
    elapsed = time.perf_counter() - t0
    print(f"End 2 end evaluation done in {elapsed:.2f}s")


def _evaluate_on_single_image(img_path, targets, split, time_predictor):
    valid_targets = [target for target in targets if target.name != "??:??"]
    with Image.open(img_path) as img:
        predictions = time_predictor.predict(img)
    targets_to_predictions = iou_bbox_matching(valid_targets, predictions)
    # TODO build pandas df
    # TODO add multiprocessing (if possible) for speedup
    evaluation_records = []
    for target, pred in targets_to_predictions.items():
        predicted_time = pred.name if pred is not None else None
        target_prediction_diff = (
            total_minutes_diff(target.name, pred.name)
            if (pred is not None and pred.name != "??:??")
            else None
        )
        record = {
            "image_path": str(img_path),
            "split": split,
            "target": target.name,
            "pred": predicted_time,
            "total_minutes_diff": target_prediction_diff,
        }
        evaluation_records.append(record)
    return evaluation_records


def load_bboxes_with_transcription(
    source: Path, split: str
) -> Tuple[List[Path], List[List[BBox]]]:
    with source.open("r") as f:
        tasks = json.load(f)
    if split is not None:
        tasks = [task for task in tasks if task["image"].startswith(split)]
    images = []
    bboxes = []
    for task in tqdm(tasks):
        image_bboxes = []
        image_transcriptions = []
        image_path = source.parent / task["image"]

        if "bbox" in task:
            for obj in task["bbox"]:
                bbox = BBox.from_ltwh(
                    obj["x"],
                    obj["y"],
                    obj["width"],
                    obj["height"],
                    obj["rectanglelabels"][0],
                ).scale(1 / 100, 1 / 100)

                image_bboxes.append(bbox)
        if "transcription" in task:
            transcriptions = task["transcription"]
            if isinstance(transcriptions, str):
                image_transcriptions.append(transcriptions)
            elif isinstance(transcriptions, list):
                image_transcriptions.extend(transcriptions)
            else:
                raise ValueError(
                    f"unknown type of transcriptions, expected list or str, "
                    f"got {type(transcriptions)}"
                )

        if len(image_bboxes) > len(image_transcriptions):
            print(f"missing transcription on task id: {task['id']}, please fix it!")
            image_transcriptions.extend(
                ["??:??"] * (len(image_transcriptions) - len(image_bboxes))
            )
        elif len(image_bboxes) < len(image_transcriptions):
            print(f"too many transcriptions on task id: {task['id']}, please fix it!")
            image_transcriptions.extend(
                ["??:??"] * (len(image_transcriptions) - len(image_bboxes))
            )

        bboxes_with_transcriptions = [
            bbox.rename(transcription)
            for bbox, transcription in zip(image_bboxes, image_transcriptions)
        ]
        bboxes.append(bboxes_with_transcriptions)
        images.append(image_path)
    return images, bboxes


if __name__ == "__main__":
    main()
