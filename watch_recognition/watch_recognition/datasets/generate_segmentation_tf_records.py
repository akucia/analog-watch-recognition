from concurrent import futures
from concurrent.futures import as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import yaml
from official.vision.data.tfrecord_lib import convert_to_feature
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from watch_recognition.datasets.generate_detection_tf_records import (
    image_info_to_feature_dict,
)
from watch_recognition.label_studio_adapters import (
    load_label_studio_polygon_detection_dataset,
)
from watch_recognition.utilities import Polygon

DATASET_SPLIT_OPTIONS = ["train", "val"]


def encode_polygons_to_label_mask(
    polygons: List[Polygon], mask_size: Tuple[int, int]
) -> np.ndarray:
    mask = np.zeros((*mask_size, 1))
    for polygon in polygons:
        if polygon.label is None:
            raise ValueError(f"polygon label is required, got {polygon.label}")
        poly_mask = polygon.to_mask(
            width=mask_size[1], height=mask_size[0], value=polygon.label
        )
        mask[:, :] = np.expand_dims(poly_mask, axis=-1)
    return mask.astype("uint8")


def polygon_annotations_to_feature_dict(polygon_annotations: List[Polygon], image_size):
    """Convert COCO annotations to an encoded feature dict."""

    feature_dict = {
        "image/segmentation/class/encoded": convert_to_feature(
            tf.io.encode_png(
                encode_polygons_to_label_mask(polygon_annotations, image_size)
            ).numpy()
        ),
        "image/segmentation/class/format": convert_to_feature(b"png")
        # "image/object/is_crowd": convert_to_feature(False),
        # TODO area
        # "image/object/area": convert_to_feature(data["area"], "float_list"),
    }

    return feature_dict


def create_tf_example(
    image_id: int,
    image_np: np.ndarray,
    polygons: List[Polygon],
):
    """Converts image and annotations to a tf.Example proto.

    Args:

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
        polygons, (image_height, image_width)
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
def main(
    input_file: str,
    output_dir: str,
    num_shards: int,
    run_concurrently: bool,
    max_images: Optional[int] = None,
    dataset_split: Optional[str] = None,
):
    dataset_path = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    label_to_cls = params["segmentation"]["label_to_cls"]
    bbox_labels = params["segmentation"]["bbox_labels"]

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
                            create_tf_example, id_, image_np, polygons
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
                tf_example = create_tf_example(id_, image_np, polygons)
                writers[idx % num_shards].write(tf_example.SerializeToString())

        for writer in writers:
            writer.close()


if __name__ == "__main__":
    main()
