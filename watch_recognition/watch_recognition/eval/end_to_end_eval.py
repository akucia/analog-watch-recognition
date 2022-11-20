import json
import os
import time
from concurrent import futures
from concurrent.futures import as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

DEFAULT_SPLITS = ["train", "val"]


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
@click.option("--run-concurrently", is_flag=True)
@click.option("--split", type=click.Choice(DEFAULT_SPLITS), default=None)
def main(run_concurrently: bool = False, split: Optional[str] = None):
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

    if split is not None:
        splits = [split]
    else:
        splits = DEFAULT_SPLITS
    print(f"Evaluating on splits: {splits}")

    records = []
    for split in splits:
        print(f"evaluating {split}")
        with source.open("r") as f:
            tasks = json.load(f)
        tasks = [task for task in tasks if task["image"].startswith(split)]
        example_ids = [task["id"] for task in tasks]
        if run_concurrently:
            print("running eval concurrently")
            with futures.ThreadPoolExecutor(os.cpu_count() // 4 or 1) as executor:
                task_futures = []
                try:
                    for example_id in example_ids:
                        future = executor.submit(
                            _evaluate_on_single_image,
                            example_id,
                            time_predictor,
                            source,
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
            for example_id in tqdm(example_ids):
                evaluation_records = _evaluate_on_single_image(
                    example_id,
                    time_predictor,
                    source,
                )
                records.extend(evaluation_records)
    df = pd.DataFrame(records)
    df = df.sort_values("image_path")
    df.to_csv("metrics/end_2_end_eval.csv", index=False)
    summary_metrics = {}
    for split in splits:
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


def _evaluate_on_single_image(
    example_id: int, time_predictor: TimePredictor, source_path: Path
) -> List[Dict]:
    targets, img_path = load_example_with_transcription(example_id, source_path)
    valid_targets = [target for target in targets if target.name != "??:??"]
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        predictions = time_predictor.predict(img)
        predictions = [
            bbox.scale(1 / img.width, 1 / img.height) for bbox in predictions
        ]
    targets_to_predictions = iou_bbox_matching(valid_targets, predictions)
    evaluation_records = []
    for target, pred in targets_to_predictions.items():
        predicted_time = pred.name if pred is not None else None
        target_prediction_diff = (
            total_minutes_diff(target.name, pred.name)
            if (pred is not None and pred.name != "??:??")
            else None
        )
        # TODO total_angle_diff
        record = {
            "example_id": example_id,
            "image_path": str(img_path),
            "split": img_path.parent.name,
            "target": target.name,
            "pred": predicted_time,
            "total_minutes_diff": target_prediction_diff,
        }
        evaluation_records.append(record)
    return evaluation_records


def load_example_with_transcription(
    example_id: int, source: Path
) -> Tuple[List[BBox], Path]:
    with source.open("r") as f:
        tasks = json.load(f)
    tasks = [task for task in tasks if task["id"] == example_id]
    if len(tasks) == 0:
        raise ValueError(f"example `{example_id}` not found")
    elif len(tasks) > 1:
        raise ValueError(f"found multiple examples for `{example_id}`")

    task = tasks[0]

    image_bboxes, image_path = _load_single_example_with_transcription(task)
    return image_bboxes, source.parent / image_path


def _load_single_example_with_transcription(task):
    image_bboxes = []
    image_transcriptions = []
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
            for entry in transcriptions:
                if isinstance(entry, str):
                    image_transcriptions.append(entry)
                else:
                    raise ValueError(
                        f"unknown type of transcriptions, expected list elemetns to be "
                        f"of type str, got {type(entry)} on task id {task['id']}"
                    )
        else:
            raise ValueError(
                f"unknown type of transcriptions, expected list or str, "
                f"got {type(transcriptions)} on task id {task['id']}"
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
    return bboxes_with_transcriptions, Path(task["image"])


if __name__ == "__main__":
    main()
