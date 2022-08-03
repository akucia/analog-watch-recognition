import json
import logging
import os
from pathlib import Path
from typing import Optional

import click
from label_studio_sdk import Client

logger = logging.getLogger(__name__)


@click.command()
@click.argument("label-studio-project")
@click.option("--label-studio-host", default=os.getenv("LABEL_STUDIO_URL"))
@click.option(
    "--label-studio-api-token", default=os.getenv("LABEL_STUDIO_ACCESS_TOKEN")
)
@click.option(
    "--export-file",
    default="watch-faces.json",
    help="Specify the directory to save dataset",
)
@click.option("--verbose", default=False)
def main(
    label_studio_project: int,
    label_studio_host: str,
    label_studio_api_token: str,
    export_file: Optional[str],
    verbose: bool,
):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    print(f"connecting to label studio at {label_studio_host}")
    ls = Client(url=label_studio_host, api_key=label_studio_api_token)
    status = ls.check_connection()
    assert status["status"] == "UP", status
    project = ls.get_project(label_studio_project)

    tasks = project.export_tasks(export_type="JSON_MIN")
    export_file = Path(export_file)
    with export_file.open("w") as f:
        json.dump(tasks, f, indent=2)


if __name__ == "__main__":
    main()
