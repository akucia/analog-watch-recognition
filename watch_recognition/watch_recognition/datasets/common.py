import hashlib
from typing import List, Tuple, Union

import numpy as np
from official.vision.data.tfrecord_lib import convert_to_feature

import tensorflow as tf
from watch_recognition.utilities import Polygon


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
        "image/object/bbox/ymin": convert_to_feature(bbox_annotations[:, 1].tolist()),
        "image/object/bbox/xmax": convert_to_feature(bbox_annotations[:, 2].tolist()),
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
    names = [p.name.encode("utf8") for p in polygon_annotations]
    feature_dict = {
        "image/segmentation/class/encoded": convert_to_feature(
            tf.io.encode_png(
                encode_polygons_to_label_mask(polygon_annotations, image_size)
            ).numpy()
        ),
        "image/segmentation/class/format": convert_to_feature(b"png"),
        "image/segmentation/class/text": convert_to_feature(names),
        # "image/object/is_crowd": convert_to_feature(False),
        # TODO area
        # "image/object/area": convert_to_feature(data["area"], "float_list"),
    }

    return feature_dict
