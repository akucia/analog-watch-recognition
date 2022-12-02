import logging
from hashlib import md5
from os import getenv
from pathlib import Path
from shutil import copy, rmtree

import click
from dotenv import load_dotenv
from google.cloud import storage
from label_studio_sdk import Client
from tqdm import tqdm

load_dotenv("../configs/local.env")


def _assign_fold_from_bytes_content(img_path):
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
        return fold


def generate_blob_gstorage_path(bucket_name: str, blob_name: str) -> str:
    blob_path = Path(bucket_name) / blob_name
    return "gs://" + str(blob_path)


def file_name_from_gstorage_url(image_url):
    image_name = image_url.split("?")[0].replace(
        "https://storage.googleapis.com/", "gs://"
    )
    return image_name


storage_client = storage.Client()


@click.command()
@click.option("--data-dir", type=click.Path(exists=True))
@click.option("--output-path", type=click.Path())
@click.option("--bucket-name", type=str, default="watch-recognition")
@click.option("--clean-save-dir", is_flag=True)
def main(data_dir, output_path, bucket_name, clean_save_dir):
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    output_path.mkdir(exist_ok=True)
    if clean_save_dir:
        rmtree(output_path)

    ls = Client(
        url=getenv("LABEL_STUDIO_URL"),
        api_key=getenv("LABEL_STUDIO_ACCESS_TOKEN"),
    )
    project = ls.get_project(int(getenv("PROJECT_ID")))
    imported_blobs = set()
    for task in project.get_tasks():
        image_url = task["data"]["image"]
        image_name = file_name_from_gstorage_url(image_url)
        imported_blobs.add(image_name)

    image_paths = list(data_dir.rglob("*.jp*g"))
    progress_bar = tqdm(image_paths)

    filtered_images = []
    for img_path in progress_bar:
        if img_path.parent.name.lower() in {"train", "test", "val"}:
            fold = img_path.parent.name.lower()
        else:
            fold = _assign_fold_from_bytes_content(img_path)
        blob_path = str(Path(fold) / img_path.name)
        blob_url = generate_blob_gstorage_path(
            bucket_name=bucket_name,
            blob_name=blob_path,
        )
        if "train" not in blob_url:
            continue
        if blob_url in imported_blobs:
            logging.debug(f"{blob_url} already in project")
            continue
        filtered_images.append((img_path, blob_path))
    print(f"{len(filtered_images)} train images left")

    for img_path, blob_path in tqdm(filtered_images):
        target_blob_path = output_path / blob_path
        target_blob_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"copy {blob_path} -> {target_blob_path}")
        copy(img_path, target_blob_path)


if __name__ == "__main__":
    main()
