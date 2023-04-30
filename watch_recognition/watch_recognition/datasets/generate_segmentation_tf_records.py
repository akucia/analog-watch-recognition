from concurrent import futures
from concurrent.futures import as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from tqdm import tqdm

from watch_recognition.datasets.common import (
    image_info_to_feature_dict,
    polygon_annotations_to_feature_dict,
)
from watch_recognition.label_studio_adapters import (
    load_label_studio_polygon_detection_dataset,
)
from watch_recognition.utilities import Polygon

DATASET_SPLIT_OPTIONS = ["train", "val"]


def _create_tf_example(
    image_id: int,
    image_np: np.ndarray,
    polygons: List[Polygon],
    border_class: int = -1,
):
    """Converts image and annotations to a tf.Example proto.

    Args:
        image_id: Unique id for the image.
        image_np: Numpy array containing the image.
        polygons: List of polygons for the image.
        border_class: Class to use for the border around the segmentation mask. If -1, no border is added.

    Returns:
      example: The converted tf.Example

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG,
        does not exist, or is not unique across image directories.
    """
    buffer = BytesIO()
    with Image.fromarray(image_np) as img:
        image_height = img.height
        image_width = img.width
        img.save(buffer, format="JPEG")
    buffer.seek(0)
    encoded_jpg = buffer.read()

    feature_dict = image_info_to_feature_dict(
        image_height, image_width, f"{image_id}.jpg", image_id, encoded_jpg, "jpg"
    )
    mask_feature_dict = polygon_annotations_to_feature_dict(
        polygons, (image_height, image_width), border_class=border_class
    )
    feature_dict.update(mask_feature_dict)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


@click.command()
@click.argument("input-file", type=str)
@click.argument("output-dir", type=str)
@click.option("--num-shards", default=10, type=int)
@click.option("--run-concurrently", is_flag=True)
@click.option("--max-images", type=int)
@click.option(
    "--dataset-split", type=click.Choice(DATASET_SPLIT_OPTIONS, case_sensitive=True)
)
@click.option(
    "--add-border-class",
    is_flag=True,
    help="Adds a border around segmentation mask as additional class",
)
def main(
    input_file: str,
    output_dir: str,
    num_shards: int,
    run_concurrently: bool,
    max_images: Optional[int] = None,
    dataset_split: Optional[str] = None,
    add_border_class: bool = False,
):
    dataset_path = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    label_to_cls = {"Hands": 1}
    bbox_labels = params["segmentation"]["bbox_labels"]
    if add_border_class:
        border_class = max(label_to_cls.values()) + 1
        print(f"Adding border class with id {border_class}")
    else:
        border_class = -1

    output_name = "watch-hands"
    if dataset_split is None:
        dataset_split = DATASET_SPLIT_OPTIONS
    else:
        dataset_split = [dataset_split]
    for split in dataset_split:
        writers = [
            tf.io.TFRecordWriter(
                str(
                    output_dir
                    / f"{output_name}-{split}-{i+1:05d}-of-{num_shards:05d}.tfrecord"
                )
            )
            for i in range(num_shards)
        ]
        dataset_gen = load_label_studio_polygon_detection_dataset(
            dataset_path,
            label_mapping=label_to_cls,
            max_num_images=max_images,
            split=split,
            bbox_labels=bbox_labels,
            crop_size=None,
        )
        if run_concurrently:
            with futures.ThreadPoolExecutor() as executor:
                task_futures = []
                try:
                    for id_, image_np, polygons in dataset_gen:
                        future = executor.submit(
                            _create_tf_example,
                            id_,
                            image_np,
                            polygons,
                            border_class,
                        )
                        task_futures.append(future)
                    for idx, future in enumerate(
                        tqdm(as_completed(task_futures), total=len(task_futures))
                    ):
                        tf_example = future.result()
                        writers[idx % num_shards].write(tf_example.SerializeToString())
                except KeyboardInterrupt:
                    print("cancelling futures")
                    for future in task_futures:
                        future.cancel()
                    for future in task_futures:
                        if not future.done():
                            print(f"waiting for {future} to complete...")
                    raise
        else:
            pbar = tqdm(dataset_gen)
            for idx, (id_, image_np, polygons) in enumerate(pbar):
                tf_example = _create_tf_example(id_, image_np, polygons, border_class)
                writers[idx % num_shards].write(tf_example.SerializeToString())

        for writer in writers:
            writer.close()


if __name__ == "__main__":
    main()
