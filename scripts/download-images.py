import json
import logging
from pathlib import Path

import click
from google.cloud import storage
from google.cloud.storage import Bucket

logger = logging.getLogger(__name__)


def download_blob(
    bucket_name: str, source_blob_name: str, destination_file: str
) -> str:
    """Uploads a file to the bucket."""
    print(
        f"downloading {source_blob_name} from {bucket_name} to {destination_file}",
        end=" ...",
    )
    save_path = Path(destination_file)
    if save_path.exists():
        print("file exists")
        return destination_file
    storage_client = storage.Client()
    bucket: Bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_file)
    print("done")
    return destination_file


@click.command()
@click.argument("dataset-file")
@click.option("--verbose", default=False)
def main(
    dataset_file: str,
    verbose: bool,
):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    dataset_file = Path(dataset_file)
    with dataset_file.open("r") as f:
        tasks = json.load(f)
    for task in tasks:
        image_gs_path = task["image"]
        split_path = image_gs_path.replace("gs://", "").split("/")
        bucket_name = split_path[0]
        file_path = "/".join(split_path[1:])
        download_path = dataset_file.parent / file_path
        download_blob(bucket_name, file_path, str(download_path))
        task["image"] = str(download_path.relative_to(dataset_file.parent))

    output_file = dataset_file.parent / (dataset_file.stem + "-local.json")

    with output_file.open("w") as f:
        json.dump(tasks, f, indent=2)


if __name__ == "__main__":
    main()
