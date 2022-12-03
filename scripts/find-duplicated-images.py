import json
import logging
from collections import defaultdict
from pathlib import Path

import click
from tqdm import tqdm

logger = logging.getLogger(__name__)


@click.command()
@click.argument("dataset-file")
@click.option("--verbose", is_flag=True)
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

    image_to_tasks = defaultdict(list)
    for task in tqdm(tasks):
        image_to_tasks[task["image"]].append(task["annotation_id"])

    duplicates = [(k, v) for k, v in image_to_tasks.items() if len(v) > 1]
    print("image url: num duplicates - annotation ids")
    for duplicate in duplicates:
        print(f"{duplicate[0]}: {len(duplicate[1])} - {duplicate[1]}")


if __name__ == "__main__":
    main()
