from functools import partial

import albumentations as A
import matplotlib.pyplot as plt
import tensorflow as tf

from watch_recognition.targets_encoding import (
    encode_keypoints_to_angle,
    encode_keypoints_to_mask,
    set_shapes,
)

DEFAULT_TRANSFORMS = A.Compose(
    [
        A.ShiftScaleRotate(
            p=0.8,
            # rotate_limit=90,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(p=0.7),
                A.RGBShift(p=0.7),
                A.ChannelShuffle(p=0.7),
                A.RandomBrightnessContrast(p=0.7),
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

# TODO deduplicate with DEFAULT_TRANSFORMS
DEFAULT_TRANSFORMS_FOR_ANGLE_CLASSIFIER = A.Compose(
    [
        A.ShiftScaleRotate(
            p=0.8,
            rotate_limit=180,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(p=0.7),
                A.RGBShift(p=0.7),
                A.ChannelShuffle(p=0.7),
                A.RandomBrightnessContrast(p=0.7),
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


def augment_kp_data(image, kp):
    image, kp = tf.numpy_function(
        func=kp_aug_fn,
        inp=[image, kp],
        Tout=(tf.uint8, tf.float32),
    )
    return image, kp


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
        fig, axarr = plt.subplots(5, masks.shape[-1] + 1, figsize=(15, 15))
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
    else:
        fig, axarr = plt.subplots(5, 2, figsize=(15, 15))
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


def get_watch_angle_dataset(
    X, y, augment: bool = True, bin_size=90, image_size=(224, 224)
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
        .shuffle(8 * 32)
        .batch(32)
        .prefetch(AUTOTUNE)
    )
    return dataset


def get_watch_keypoints_dataset(
    X, y, augment: bool = True, batch_size=32, image_size=None, mask_size=None
) -> tf.data.Dataset:
    encode_kp = partial(
        encode_keypoints_to_mask,
        image_size=(*image_size, 3),
        mask_size=mask_size,
        radius=1,
        include_background=False,
        separate_hour_and_minute_hands=False,
        add_perimeter=True,
        with_perimeter_for_hands=False,
        sparse=False,
        blur=True,
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

    dataset = (
        dataset.map(encode_kp, num_parallel_calls=AUTOTUNE)
        .map(set_shape_f, num_parallel_calls=AUTOTUNE)
        .shuffle(8 * batch_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return dataset
