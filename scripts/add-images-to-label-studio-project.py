import dataclasses
import json
import logging
import tarfile
from hashlib import md5
from pathlib import Path
from random import shuffle
from typing import Optional

import click
import requests
from google.cloud import storage
from google.cloud.storage import Bucket
from label_studio_sdk import Client
from PIL import Image
from tqdm import tqdm

from watch_recognition.models import points_to_time
from watch_recognition.predictors import (
    HandPredictor,
    KPHeatmapPredictorV2,
    RetinanetDetector,
)

logger = logging.getLogger(__name__)


def upload_blob(
    bucket_name: str, source_file_name: str, destination_blob_name: str
) -> str:
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket: Bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        logger.debug(f"object {destination_blob_name} already exists - skipping")
        return destination_blob_name

    blob.upload_from_filename(source_file_name)

    logger.debug(f"File {source_file_name} uploaded to {destination_blob_name}.")
    return destination_blob_name


def generate_blob_gstorage_path(bucket_name: str, blob_name: str) -> str:
    blob_path = Path(bucket_name) / blob_name
    return "gs://" + str(blob_path)


def download_and_uzip_model(url: str, save_dir: str = "/tmp/") -> Path:
    save_dir = Path(save_dir)
    name = url.split("/")[-1]
    save_file = save_dir / name
    if not save_file.exists():
        logger.debug(f"downloading {name}")
        with requests.get(url, stream=True) as response:
            with save_file.open("wb") as f:
                f.write(response.content)
    extract_dir = save_dir / save_file.stem
    with tarfile.open(save_file) as tar:
        logger.debug(f"extracting {name}")
        tar.extractall(extract_dir)

    return extract_dir


@click.command()
@click.option(
    "--source-dir",
    help="Specify the source images directory",
)
@click.option(
    "--export-file",
    default=None,
    help="Specify the directory to save dataset",
)
@click.option(
    "--bucket-name",
    default="watch-recognition",
    help="Specify the GStorage bucket tu upload images",
)
@click.option("--verbose", default=False)
@click.option("--label-studio-project")
@click.option("--label-studio-host")
@click.option("--label-studio-api-token")
@click.option("--n-images", help="Number of images to add", type=int)
@click.option("--shuffle-images", default=False, type=bool)
def main(
    source_dir: str,
    export_file: Optional[str],
    bucket_name: str,
    verbose: bool,
    label_studio_project: int,
    label_studio_host: str,
    label_studio_api_token: str,
    n_images: int,
    shuffle_images: bool,
):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    source_dir = Path(source_dir)
    assert source_dir.exists()

    ls = Client(url=label_studio_host, api_key=label_studio_api_token)
    project = ls.get_project(label_studio_project)
    imported_blobs = set()
    for task in project.get_tasks():
        image_url = task["data"]["image"]
        image_name = file_name_from_gstorage_url(image_url)
        imported_blobs.add(image_name)

    # project.connect_google_import_storage(
    #     bucket=bucket_name,
    #     google_application_credentials=service_account_file,
    #     presign_ttl=60,
    #     title=bucket_name,
    # )
    cls_to_label = {0: "WatchFace"}
    detector = RetinanetDetector(
        Path("models/detector/"), class_to_label_name=cls_to_label
    )

    hand_predictor = HandPredictor(
        download_and_uzip_model(
            url="https://storage.googleapis.com/akuc-ml-public/models/effnet-b3-FPN-160-tversky-hands.tar.gz"
        )
    )
    kp_predictor = KPHeatmapPredictorV2(
        "models/keypoint",
        class_to_label_name={
            "Top": 0,
            "Center": 1,
            "Crown": 2,
        },
        confidence_threshold=0.5,
    )

    dataset = []
    image_paths = list(source_dir.rglob("*.jp*g"))
    if shuffle_images:
        shuffle(image_paths)
    if n_images:
        image_paths = image_paths[:n_images]
    progress_bar = tqdm(image_paths)
    for img_path in progress_bar:
        progress_bar.set_description(img_path.name)
        if img_path.parent.name.lower() in {"train", "test", "val"}:
            fold = img_path.parent.name.lower()
        else:
            with img_path.open("rb") as f:
                data = f.read()
                int_hash = int(md5(data).hexdigest(), 16)
                float_hash = int_hash / 2**128
            if float_hash < 0.8:
                fold = "train"
            elif float_hash < 0.9:
                fold = "val"
            else:
                fold = "test"
        blob_path = str(Path(fold) / img_path.name)
        blob_url = generate_blob_gstorage_path(
            bucket_name=bucket_name,
            blob_name=blob_path,
        )
        if blob_url in imported_blobs:
            continue
        upload_blob(
            bucket_name=bucket_name,
            source_file_name=img_path,
            destination_blob_name=blob_path,
        )

        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            bboxes = detector.predict(pil_img)

            bboxes = [dataclasses.replace(bbox, name="WatchFace") for bbox in bboxes]
            polygons = []
            keypoints = []
            transcriptions = []
            for box in bboxes:
                points = kp_predictor.predict_from_image_and_bbox(pil_img, box)
                pred_center = [p for p in points if p.name == "Center"]
                if pred_center:
                    pred_center = pred_center[0]
                else:
                    # if there's no center available - skip the bbox
                    continue

                pred_top = [p for p in points if p.name == "Top"]
                if pred_top:
                    pred_top = pred_top[0]
                (
                    minute_and_hour,
                    other,
                    polygon,
                ) = hand_predictor.predict_from_image_and_bbox(
                    pil_img, box, pred_center, debug=False
                )
                polygons.append(polygon)
                if minute_and_hour:
                    pred_minute, pred_hour = minute_and_hour
                    minute_kp = dataclasses.replace(pred_minute.end, name="Minute")
                    hour_kp = dataclasses.replace(pred_hour.end, name="Hour")
                    read_hour, read_minute = points_to_time(
                        pred_center, hour_kp, minute_kp, pred_top
                    )

                    time = f"{read_hour:02.0f}:{read_minute:02.0f}"
                    transcriptions.append(time)
                else:
                    transcriptions.append("??:??")
            print(transcriptions)
            # TODO add Transcriptions to results
            results = []
            # The code below adds them as a separate rectangle,
            # which is not ideal and won't allow for edits
            # for i, (transcription, bbox) in enumerate(zip(transcriptions, bboxes)):
            #     value = bbox.as_label_studio_object
            #     value["text"] = [transcription]
            #     del value['rectanglelabels']
            #     results.append(
            #         {
            #             "value": value,
            #             "to_name": "image",
            #             "from_name": "transcription",
            #             "type": "textarea",
            #             "id": i,
            #         }
            #     )
            for bbox in bboxes:
                results.append(
                    {
                        "value": bbox.scale(
                            1 / pil_img.width, 1 / pil_img.height
                        ).as_label_studio_object,
                        "to_name": "image",
                        "from_name": "bbox",
                        "type": "rectanglelabels",
                    }
                )
            polygons = [
                dataclasses.replace(polygon, name="Hands") for polygon in polygons
            ]
            for polygon in polygons:
                results.append(
                    {
                        "value": polygon.scale(
                            1 / pil_img.width, 1 / pil_img.height
                        ).as_label_studio_object,
                        "to_name": "image",
                        "from_name": "polygon",
                        "type": "polygonlabels",
                    }
                )
            for kp in keypoints:
                results.append(
                    {
                        "value": kp.scale(
                            1 / pil_img.width, 1 / pil_img.height
                        ).as_label_studio_object,
                        "to_name": "image",
                        "from_name": "kp",
                        "type": "keypointlabels",
                    }
                )
            image_data = {
                "data": {
                    "image": blob_url,
                },
                "predictions": [{"result": results}],
            }
            dataset.append(image_data)
            project.import_tasks([image_data])
    if export_file:
        with Path(export_file).open("w") as f:
            json.dump(dataset, f, indent=2)


def file_name_from_gstorage_url(image_url):
    image_name = image_url.split("?")[0].replace(
        "https://storage.googleapis.com/", "gs://"
    )
    return image_name


if __name__ == "__main__":
    main()
