import hashlib
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from official.vision.data.tfrecord_lib import convert_to_feature
from skimage.morphology import dilation, square

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


def generate_border(
    mask: np.ndarray, border_class: int = 1, border_width: int = 1
) -> np.ndarray:
    binary_mask = (mask > 0).astype("uint8").squeeze()
    footprint = square(border_width)
    border = (dilation(binary_mask, footprint=footprint) - binary_mask) > 0
    mask[border] = border_class
    return mask


def encode_polygons_to_label_mask(
    polygons: List[Polygon], mask_size: Tuple[int, int], border_class: int = -1
) -> np.ndarray:
    mask = np.zeros((*mask_size, 1))
    for polygon in polygons:
        if polygon.label is None:
            raise ValueError(f"polygon label is required, got {polygon.label}")
        poly_mask = polygon.to_mask(width=mask_size[1], height=mask_size[0])
        poly_mask = np.where(poly_mask, polygon.label, 0)
        if border_class > 0:
            border_width = np.sqrt(mask_size[0] * mask_size[1]).astype(int) // 100
            border_width = max(border_width, 3)
            poly_mask = generate_border(
                poly_mask,
                border_class,
                border_width=border_width,
            )
        mask[:, :] = np.expand_dims(poly_mask, axis=-1)
    return mask.astype("uint8")


def polygon_annotations_to_feature_dict(
    polygon_annotations: List[Polygon],
    image_size: Tuple[int, int],
    border_class: int = -1,
):
    """Convert COCO annotations to an encoded feature dict."""
    feature_dict = {
        "image/segmentation/class/encoded": convert_to_feature(
            tf.io.encode_png(
                encode_polygons_to_label_mask(
                    polygon_annotations, image_size, border_class
                )
            ).numpy()
        ),
        "image/segmentation/class/format": convert_to_feature(b"png"),
    }

    return feature_dict
