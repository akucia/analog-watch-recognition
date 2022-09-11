import tensorflow as tf
from keras_cv import bounding_box


def _convert_format_and_resize_image(bounding_box_format, img_size):
    """Mapping function to create batched image and bbox coordinates
    Based on Keras-cv example:
    https://github.com/keras-team/keras-cv/blob/8b8734d633e423ab70bc23bf9cc02d7aa6003ed4/keras_cv/datasets/pascal_voc/load.py#L48
    """

    resizing = tf.keras.layers.Resizing(
        height=img_size[0], width=img_size[1], crop_to_aspect_ratio=False
    )

    def apply(inputs):
        inputs["image"] = resizing(inputs["image"])
        inputs["objects"]["bbox"] = bounding_box.convert_format(
            inputs["objects"]["bbox"],
            images=inputs["image"],
            source="rel_xyxy",
            target=bounding_box_format,
        )

        bounding_boxes = inputs["objects"]["bbox"]
        labels = tf.cast(inputs["objects"]["label"], tf.float32)
        labels = tf.expand_dims(labels, axis=-1)
        bounding_boxes = tf.concat([bounding_boxes, labels], axis=-1)
        return {"images": inputs["image"], "bounding_boxes": bounding_boxes}

    return apply


def _decode_detection_ds_fn(example):
    """"""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "objects/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "objects/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "objects/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "objects/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "objects/bbox/label": tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    xmin = tf.sparse.to_dense(example["objects/bbox/xmin"])
    ymin = tf.sparse.to_dense(example["objects/bbox/ymin"])
    xmax = tf.sparse.to_dense(example["objects/bbox/xmax"])
    ymax = tf.sparse.to_dense(example["objects/bbox/ymax"])
    bbox = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return {
        "image": image,
        "objects": {
            "bbox": bbox,
            "label": tf.sparse.to_dense(example["objects/bbox/label"]),
        },
    }


def load(
    filenames,
    bounding_box_format,
    batch_size=None,
    shuffle=True,
    shuffle_buffer=None,
    img_size=(512, 512),
):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_decode_detection_ds_fn)
    _map_fn = _convert_format_and_resize_image(
        bounding_box_format=bounding_box_format, img_size=img_size
    )

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        if not batch_size and not shuffle_buffer:
            raise ValueError(
                "If `shuffle=True`, either a `batch_size` or `shuffle_buffer` must be "
                "provided to `keras_cv.datasets.pascal_voc.load().`"
            )
        shuffle_buffer = shuffle_buffer or 8 * batch_size
    dataset = dataset.shuffle(shuffle_buffer)
    if batch_size is not None:
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)
        )
    return dataset, None
