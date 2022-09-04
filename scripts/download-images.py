import json
import logging
import time
from concurrent import futures
from concurrent.futures import as_completed
from functools import partial
from pathlib import Path

import click
import google
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import storage
from google.cloud.storage import Bucket
from tqdm import tqdm

logger = logging.getLogger(__name__)

storage_client = storage.Client()


def download_blob(
    bucket_name: str, source_blob_name: str, destination_file: str, force_download: bool
) -> str:
    """Uploads a file to the bucket."""
    save_path = Path(destination_file)
    if save_path.exists() and not force_download:
        # print(
        #     f"skipping download {source_blob_name} from {bucket_name} to {destination_file} - file exists"
        # )
        return destination_file
    bucket: Bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_file)
    # print(f"downloaded {source_blob_name} from {bucket_name} to {destination_file}")
    return destination_file


@click.command()
@click.argument("dataset-file")
@click.option("--verbose", is_flag=True)
@click.option("--concurrent", is_flag=True)
@click.option("--force-download", is_flag=True)
def main(
    dataset_file: str,
    verbose: bool,
    concurrent: bool,
    force_download: bool,
):
    t0 = time.perf_counter()
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    dataset_file = Path(dataset_file)
    with dataset_file.open("r") as f:
        tasks = json.load(f)
    dataset_dir = dataset_file.parent
    _process_single_task = partial(
        _download_image_and_convert_path,
        dataset_dir=dataset_dir,
        force_download=force_download,
    )

    if concurrent:
        new_tasks = []
        with futures.ThreadPoolExecutor() as executor:
            task_futures = []
            try:
                for task in tasks:
                    future = executor.submit(
                        _download_image_and_convert_path,
                        task,
                        dataset_dir=dataset_dir,
                        force_download=force_download,
                    )
                    task_futures.append(future)
                for future in tqdm(as_completed(task_futures), total=len(task_futures)):
                    try:
                        new_tasks.append(future.result())
                    except GoogleAPICallError as e:
                        print(e)
            except KeyboardInterrupt:
                print("cancelling futures")
                for future in task_futures:
                    future.cancel()
                for future in task_futures:
                    if not future.done():
                        print(f"waiting for {future} to complete...")
                raise
    else:
        new_tasks = []
        for task in tqdm(tasks):
            new_tasks.append(
                _download_image_and_convert_path(task, dataset_dir, force_download)
            )

    new_tasks = sorted(new_tasks, key=lambda task: task["id"], reverse=True)

    output_file = dataset_dir / (dataset_file.stem + "-local.json")

    with output_file.open("w") as f:
        json.dump(new_tasks, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"Images downloaded in {elapsed:.2f}s")


def _download_image_and_convert_path(task, dataset_dir, force_download):
    split_path = task["image"].replace("gs://", "").split("/")
    bucket_name = split_path[0]
    file_path = "/".join(split_path[1:])
    download_path = dataset_dir / file_path
    download_blob(bucket_name, file_path, str(download_path), force_download)
    new_image_path = str(download_path.relative_to(dataset_dir))
    task["image"] = new_image_path
    return task


if __name__ == "__main__":
    main()
