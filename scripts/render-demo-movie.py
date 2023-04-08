#!/usr/bin/env python

import concurrent
import itertools
from pathlib import Path
from threading import Event, Thread
from typing import Tuple

import click
import cv2
import numpy as np
from tqdm import tqdm

from watch_recognition.predictors import (
    HandPredictorGRPC,
    KPHeatmapPredictorV2GRPC,
    RetinaNetDetectorGRPC,
    TimePredictor,
)

kp_name_to_color = {
    "Top": (255, 0, 0),
    "Center": (0, 255, 0),
    "Crown": (0, 0, 255),
}
line_name_to_color = {
    "Hour": (255, 0, 0),
    "Minute": (0, 255, 0),
}


def spin(msg: str, done: Event) -> None:
    for char in itertools.cycle(r"\|/-"):
        status = f"\r{char} {msg}"
        print(status, end="", flush=True)
        if done.wait(0.1):
            break
    blanks = " " * len(status)
    print(f"\r{blanks}\r", end="")


@click.command()
@click.argument("input-file", type=click.Path(exists=True))
@click.argument("output-file", type=click.Path())
@click.option("--max-frames", default=-1, type=int)
@click.option("--enable-multithreading", is_flag=True)
def main(
    input_file: str,
    output_file: str,
    max_frames: int = -1,
    enable_multithreading: bool = False,
):
    print(f"Running predictions on {input_file} and saving to {output_file}.")
    file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    predictor = TimePredictor(
        detector=RetinaNetDetectorGRPC(
            "localhost:8500",
            "detector",
            class_to_label_name={1: "WatchFace"},
        ),
        kp_predictor=KPHeatmapPredictorV2GRPC(
            "localhost:8500",
            "keypoint",
            class_to_label_name={
                0: "Top",
                1: "Center",
                2: "Crown",
            },
            confidence_threshold=0.5,
        ),
        hand_predictor=HandPredictorGRPC("localhost:8500", "hands"),
    )
    cap = cv2.VideoCapture(str(file))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = int(cap.get(cv2.CAP_PROP_FPS))

    crop_padding = (frame_height - frame_width) // 2
    if enable_multithreading:
        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            futures = []
            done = Event()
            spinner = Thread(target=spin, args=("loading input frames...", done))
            spinner.start()
            while cap.isOpened():
                if 0 < max_frames <= len(futures):
                    break

                success, frame = cap.read()
                if not success:
                    break

                future = executor.submit(
                    _process_frame, crop_padding, frame, predictor, len(futures)
                )
                futures.append(future)

            done.set()
            spinner.join()
            print("frames loaded")
            processed_frames = []
            futures_iterator = concurrent.futures.as_completed(futures)
            for future in tqdm(futures_iterator, total=len(futures)):
                processed_frames.append(future.result())
    else:
        processed_frames = []
        pbar = tqdm()
        while cap.isOpened():
            if 0 < max_frames <= len(processed_frames):
                break
            success, frame = cap.read()
            if not success:
                break
            processed_frames.append(
                _process_frame(crop_padding, frame, predictor, len(processed_frames))
            )
            pbar.update(1)
        pbar.close()
    cap.release()

    print(f"saving {len(processed_frames)} frames")
    processed_frames = sorted(processed_frames, key=lambda x: x[0])

    out = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps - 10,
        (int(frame_width), int(frame_width)),  # square video
    )
    for frame in processed_frames:
        out.write(frame[1])
    out.release()
    cv2.destroyAllWindows()


def _process_frame(
    crop_padding: int, frame: np.ndarray, predictor: TimePredictor, frame_id: int
) -> Tuple[int, np.ndarray]:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame[crop_padding:-crop_padding, :]
    frame = predictor.predict_and_draw(frame, kp_name_to_color, line_name_to_color)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_id, frame


if __name__ == "__main__":
    main()
