from functools import partial

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from watch_recognition.targets_encoding import (
    add_sample_weights,
    encode_keypoints_to_angle,
    encode_keypoints_to_hands_angles,
    encode_keypoints_to_mask,
    set_shapes,
    set_shapes_with_sample_weight,
)

EMPTY_TRANSFORMS = A.Compose(
    [],
    # format="xyas" is required while using tf.Data pipielines, otherwise
    # tf cannot validate the output shapes # TODO check if this is correct
    # remove_invisible=False is required to preserve the order and number of keypoints
    keypoint_params=A.KeypointParams(format="xyas", remove_invisible=False),
)

DEFAULT_TRANSFORMS = A.Compose(
    [
        A.OneOf(
            [
                A.HueSaturationValue(),
                A.RGBShift(),
                A.ChannelShuffle(),
            ],
            p=1,
        ),
        # A.MotionBlur(),
    ],
    # format="xyas" is required while using tf.Data pipielines, otherwise
    # tf cannot validate the output shapes # TODO check if this is correct
    # remove_invisible=False is required to preserve the order and number of keypoints
    keypoint_params=A.KeypointParams(format="xyas", remove_invisible=False),
)

DEFAULT_TRANSFORMS_FOR_MASKS = A.Compose(
    [
        # A.HorizontalFlip(),
        # A.VerticalFlip(),
        A.RandomRotate90(),
        A.RandomSizedCrop(
            (120, 160),  # TODO min,max size should be passes as params
            160,  # timage size should be passed as params to generate dataset
            160,
            interpolation=cv2.INTER_CUBIC,
            p=0.5,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(p=0.8),
                A.RGBShift(p=0.8),
                A.ChannelShuffle(p=0.8),
                A.ChannelDropout(p=0.8),
                A.CLAHE(p=0.8),
                A.ISONoise(p=0.8),
                A.InvertImg(p=0.8),
            ],
            p=1,
        ),
    ],
)


# TODO deduplicate with DEFAULT_TRANSFORMS
DEFAULT_TRANSFORMS_FOR_ANGLE_CLASSIFIER = A.Compose(
    [
        A.ShiftScaleRotate(
            p=0.8,
            rotate_limit=180,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(p=1),
                A.RGBShift(p=1),
                A.ChannelShuffle(p=1),
                A.RandomBrightnessContrast(p=1),
            ],
            p=1,
        ),
        A.MotionBlur(),
    ],
    # format="xyas" is required while using tf.Data pipielines, otherwise
    # tf cannot validate the output shapes # TODO check if this is correct
    # remove_invisible=False is required to preserve the order and number of keypoints
    keypoint_params=A.KeypointParams(format="xyas", remove_invisible=False),
)


def kp_aug_fn(image, keypoints):
    data = {
        "image": image,
        "keypoints": keypoints,
    }
    aug_data = DEFAULT_TRANSFORMS(**data)
    aug_img = aug_data["image"]
    aug_kp = aug_data["keypoints"]
    aug_kp = tf.cast(aug_kp, tf.float32)
    return aug_img, aug_kp


def kp_angle_fn(image, keypoints):
    data = {
        "image": image,
        "keypoints": keypoints,
    }
    aug_data = DEFAULT_TRANSFORMS_FOR_ANGLE_CLASSIFIER(**data)
    aug_img = aug_data["image"]
    aug_kp = aug_data["keypoints"]
    aug_kp = tf.cast(aug_kp, tf.float32)
    return aug_img, aug_kp


def mask_aug_fn(image, mask):
    data = {
        "image": image,
        "mask": mask,
    }
    aug_data = DEFAULT_TRANSFORMS_FOR_MASKS(**data)
    aug_img = aug_data["image"]
    aug_mask = aug_data["mask"]
    aug_mask = tf.cast(aug_mask, tf.float32)
    return aug_img, aug_mask


def augment_kp_data(image, kp):
    image, kp = tf.numpy_function(
        func=kp_aug_fn,
        inp=[image, kp],
        Tout=(tf.uint8, tf.float32),
    )
    return image, kp


def augment_mask_data(image, mask):
    image, mask = tf.numpy_function(
        func=mask_aug_fn,
        inp=[image, mask],
        Tout=(tf.uint8, tf.float32),
    )
    return image, mask


def augment_kp_angle_cls_data(image, kp):
    image, kp = tf.numpy_function(
        func=kp_angle_fn,
        inp=[image, kp],
        Tout=(tf.uint8, tf.float32),
    )
    return image, kp


def view_image(ds):

    batch = next(iter(ds))  # extract 1 batch from the dataset
    image, masks = batch[0], batch[1]
    image = image.numpy()
    masks = masks.numpy()
    if masks.shape[-1] > 1:
        fig, axarr = plt.subplots(5, masks.shape[-1] + 2, figsize=(15, 15))
        for i in range(5):
            ax = axarr[i]
            img = image[i]
            ax_idx = 0
            ax[ax_idx].imshow(img.astype("uint8"))
            ax[ax_idx].set_xticks([])
            ax[ax_idx].set_yticks([])
            ax[ax_idx].set_title("Image")
            for j in range(masks.shape[-1]):
                ax_idx = j + 1
                ax[ax_idx].imshow(masks[i, :, :, j])
                ax[ax_idx].set_title("Masks")

            merged = cv2.addWeighted(
                img, 0.5, (masks[i] * 255).astype("uint8"), 0.5, 0.0
            )

            ax_idx += 1
            ax[ax_idx].imshow(merged)
            ax[ax_idx].set_title("Masks + Image")
    else:
        fig, axarr = plt.subplots(5, 3, figsize=(15, 15))
        for i in range(5):
            ax = axarr[i]
            img = image[i]
            ax_idx = 0
            ax[ax_idx].imshow(img.astype("uint8"))
            ax[ax_idx].set_xticks([])
            ax[ax_idx].set_yticks([])
            ax[ax_idx].set_title("Image")

            ax_idx += 1
            ax[ax_idx].imshow(masks[i, :, :, -1].astype("uint8"))
            ax[ax_idx].set_title("Masks")

            merged = cv2.addWeighted(
                img, 0.5, (masks[i, :, :, -1] * 255).astype("uint8"), 0.5, 0.0
            )

            ax_idx += 1
            ax[ax_idx].imshow(merged)
            ax[ax_idx].set_title("Masks + Image")


def get_watch_angle_dataset(
    X, y, augment: bool = True, bin_size=90, image_size=(224, 224), batch_size=32
) -> tf.data.Dataset:
    encode_kp = partial(encode_keypoints_to_angle, bin_size=bin_size)
    set_shape_f = partial(
        set_shapes, img_shape=(*image_size, 3), target_shape=(360 // bin_size,)
    )

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if augment:
        dataset = dataset.map(
            augment_kp_angle_cls_data,
            num_parallel_calls=AUTOTUNE,
        )
    dataset = (
        dataset.map(encode_kp, num_parallel_calls=AUTOTUNE)
        .map(set_shape_f, num_parallel_calls=AUTOTUNE)
        .shuffle(8 * batch_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset


def get_hands_angles_dataset(
    X, y, augment: bool = True, image_size=(96, 96), batch_size=32
) -> tf.data.Dataset:
    encode_kp = partial(encode_keypoints_to_hands_angles)
    set_shape_f = partial(set_shapes, img_shape=(*image_size, 3), target_shape=(2,))

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if augment:
        dataset = dataset.map(
            augment_kp_angle_cls_data,
            num_parallel_calls=AUTOTUNE,
        )
    dataset = (
        dataset.map(encode_kp, num_parallel_calls=AUTOTUNE)
        .map(set_shape_f, num_parallel_calls=AUTOTUNE)
        .shuffle(8 * batch_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset


def get_watch_hands_mask_dataset(
    X, y, augment: bool = True, image_size=(224, 224), batch_size=32, class_weights=None
) -> tf.data.Dataset:

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if augment:
        dataset = dataset.map(
            augment_mask_data,
            num_parallel_calls=AUTOTUNE,
        )

    if class_weights:
        add_sample_weights_f = partial(add_sample_weights, class_weights=class_weights)
        dataset = dataset.map(add_sample_weights_f)
        set_shape_f = partial(
            set_shapes_with_sample_weight,
            img_shape=(*image_size, 3),
            target_shape=(*image_size, 1),
        )
    else:
        set_shape_f = partial(
            set_shapes, img_shape=(*image_size, 3), target_shape=(*image_size, 1)
        )
    dataset = (
        dataset.map(set_shape_f, num_parallel_calls=AUTOTUNE)
        .shuffle(8 * batch_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset


def get_watch_keypoints_dataset(
    X,
    y,
    augment: bool = True,
    batch_size=32,
    image_size=None,
    mask_size=None,
    shuffle=True,
) -> tf.data.Dataset:
    encode_kp = partial(
        encode_keypoints_to_mask,
        image_size=(*image_size, 3),
        mask_size=mask_size,
        radius=4,
        include_background=False,
        separate_hour_and_minute_hands=False,
        add_perimeter=True,
        with_perimeter_for_hands=True,
        sparse=False,
        blur=True,
        hands_as_lines=True,
    )
    set_shape_f = partial(
        set_shapes, img_shape=(*image_size, 3), target_shape=(*mask_size, 3)
    )
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if augment:
        dataset = dataset.map(
            augment_kp_data,
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.map(encode_kp, num_parallel_calls=AUTOTUNE).map(
        set_shape_f, num_parallel_calls=AUTOTUNE
    )
    if shuffle:
        dataset = dataset.shuffle(8 * batch_size)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
