import hashlib
from concurrent import futures
from concurrent.futures import as_completed
from pathlib import Path
from typing import Dict, Optional, Union

import click
import numpy as np
import yaml
from official.vision.data.tfrecord_lib import convert_to_feature
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from watch_recognition.label_studio_adapters import (
    load_label_studio_bbox_detection_dataset,
)


def image_info_to_feature_dict(
    height: int,
    width: int,
    filename: str,
    image_id: Union[str, int],
    encoded_str: bytes,
    encoded_format: str,
):
    """Convert image information to a dict of features."""

    key = hashlib.sha256(encoded_str).hexdigest()

    return {
        "image/height": convert_to_feature(height),
        "image/width": convert_to_feature(width),
        "image/filename": convert_to_feature(filename.encode("utf8")),
        "image/source_id": convert_to_feature(str(image_id).encode("utf8")),
        "image/key/sha256": convert_to_feature(key.encode("utf8")),
        "image/encoded": convert_to_feature(encoded_str),
        "image/format": convert_to_feature(encoded_format.encode("utf8")),
    }


def bbox_annotations_to_feature_dict(
    bbox_annotations, class_annotations, id_to_name_map
):
    """Convert COCO annotations to an encoded feature dict."""

    names = [
        id_to_name_map[name].encode("utf8")
        for name in class_annotations.flatten().tolist()
    ]
    feature_dict = {
        "image/object/bbox/xmin": convert_to_feature(bbox_annotations[:, 0].tolist()),
        "image/object/bbox/xmax": convert_to_feature(bbox_annotations[:, 2].tolist()),
        "image/object/bbox/ymin": convert_to_feature(bbox_annotations[:, 1].tolist()),
        "image/object/bbox/ymax": convert_to_feature(bbox_annotations[:, 3].tolist()),
        "image/object/class/text": convert_to_feature(names),
        "image/object/class/label": convert_to_feature(
            class_annotations.flatten().tolist()
        ),
        # "image/object/is_crowd": convert_to_feature(False),
        # TODO area
        # "image/object/area": convert_to_feature(data["area"], "float_list"),
    }

    return feature_dict


def mask_annotations_to_feature_dict(
    bbox_annotations, class_annotations, id_to_name_map
):
    """Convert COCO annotations to an encoded feature dict."""

    names = [
        id_to_name_map[name].encode("utf8")
        for name in class_annotations.flatten().tolist()
    ]
    feature_dict = {
        "image/object/bbox/xmin": convert_to_feature(bbox_annotations[:, 0].tolist()),
        "image/object/bbox/xmax": convert_to_feature(bbox_annotations[:, 2].tolist()),
        "image/object/bbox/ymin": convert_to_feature(bbox_annotations[:, 1].tolist()),
        "image/object/bbox/ymax": convert_to_feature(bbox_annotations[:, 3].tolist()),
        "image/object/class/text": convert_to_feature(names),
        "image/object/class/label": convert_to_feature(
            class_annotations.flatten().tolist()
        ),
        # "image/object/is_crowd": convert_to_feature(False),
        # TODO area
        # "image/object/area": convert_to_feature(data["area"], "float_list"),
    }

    return feature_dict


def create_tf_example(
    image_id: int,
    image_path: Path,
    bboxes: np.ndarray,
    classes: np.ndarray,
    id_to_name_map: Dict[int, str],
):
    """Converts image and annotations to a tf.Example proto.

    Args:

    Returns:
      example: The converted tf.Example

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG,
        does not exist, or is not unique across image directories.
    """
    filename = image_path.name

    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_jpg = fid.read()

    with Image.open(image_path) as img:
        image_height = img.height
        image_width = img.width

    feature_dict = image_info_to_feature_dict(
        image_height, image_width, filename, image_id, encoded_jpg, "jpg"
    )
    if len(classes):
        box_feature_dict = bbox_annotations_to_feature_dict(
            bboxes, classes, id_to_name_map
        )
        feature_dict.update(box_feature_dict)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


DATASET_SPLIT_OPTIONS = ["train", "val", "test"]


@click.command()
@click.argument("input-file", type=str)
@click.argument("output-dir", type=str)
@click.option("--num-shards", default=10, type=int)
@click.option("--run-concurrently", is_flag=True)
@click.option("--max-images", type=int)
@click.option(
    "--dataset-split", type=click.Choice(DATASET_SPLIT_OPTIONS, case_sensitive=True)
)
@click.option("--skip-images-without-annotations", type=bool, is_flag=True)
def main(
    input_file: str,
    output_dir: str,
    num_shards: int,
    run_concurrently: bool,
    max_images: Optional[int] = None,
    dataset_split: Optional[str] = None,
    skip_images_without_annotations: bool = False,
):
    dataset_path = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    label_to_cls = params["detector"]["label_to_cls"]

    cls_to_label = {v: k for k, v in label_to_cls.items()}
    output_name = "watch-faces"
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
        dataset_gen = load_label_studio_bbox_detection_dataset(
            dataset_path,
            label_mapping=label_to_cls,
            split=split,
            skip_images_without_annotations=skip_images_without_annotations,
            max_num_images=max_images,
        )
        if run_concurrently:
            with futures.ThreadPoolExecutor() as executor:
                task_futures = []
                try:
                    for id_, image_path, bboxes, classes in dataset_gen:
                        future = executor.submit(
                            create_tf_example,
                            id_,
                            image_path,
                            bboxes,
                            classes,
                            cls_to_label,
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
            for idx, (id_, image_path, bboxes, classes) in enumerate(pbar):
                tf_example = create_tf_example(
                    id_, image_path, bboxes, classes, cls_to_label
                )
                writers[idx % num_shards].write(tf_example.SerializeToString())

        for writer in writers:
            writer.close()


if __name__ == "__main__":
    main()
