import logging
import tarfile
from pathlib import Path

import click
import requests
from google.cloud import storage
from google.cloud.storage import Bucket
from label_studio_sdk import Client

storage_client = storage.Client()


def upload_blob(
    bucket_name: str, source_file_name: str, destination_blob_name: str
) -> str:
    """Uploads a file to the bucket."""

    bucket: Bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        logging.debug(f"object {destination_blob_name} already exists - skipping")
        return destination_blob_name

    blob.upload_from_filename(source_file_name)

    logging.debug(f"File {source_file_name} uploaded to {destination_blob_name}.")
    return destination_blob_name


def generate_blob_gstorage_path(bucket_name: str, blob_name: str) -> str:
    blob_path = Path(bucket_name) / blob_name
    return "gs://" + str(blob_path)


def download_and_uzip_model(url: str, save_dir: str = "/tmp/") -> Path:
    save_dir = Path(save_dir)
    name = url.split("/")[-1]
    save_file = save_dir / name
    if not save_file.exists():
        logging.debug(f"downloading {name}")
        with requests.get(url, stream=True) as response:
            with save_file.open("wb") as f:
                f.write(response.content)
    extract_dir = save_dir / save_file.stem
    with tarfile.open(save_file) as tar:
        logging.debug(f"extracting {name}")
        tar.extractall(extract_dir)

    return extract_dir


@click.command()
@click.option(
    "--bucket-name",
    default="watch-recognition",
    help="Specify the GStorage bucket tu upload images",
)
@click.option("--label-studio-project")
@click.option("--label-studio-host")
@click.option("--label-studio-api-token")
@click.option("--service-account-file")
def main(
    bucket_name: str,
    label_studio_project: int,
    label_studio_host: str,
    label_studio_api_token: str,
    service_account_file: str,
):

    ls = Client(url=label_studio_host, api_key=label_studio_api_token)
    project = ls.get_project(label_studio_project)

    project.connect_google_import_storage(
        bucket=bucket_name,
        google_application_credentials=service_account_file,
        presign_ttl=60,
        title=bucket_name,
    )


if __name__ == "__main__":
    main()
