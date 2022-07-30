from functools import partial

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
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
        A.Rotate(limit=30, p=0.8),
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
        # A.MotionBlur(),
    ],
    # format="xyas" is required while using tf.Data pipielines, otherwise
    # tf cannot validate the output shapes # TODO check if this is correct
    # remove_invisible=False is required to preserve the order and number of keypoints
    keypoint_params=A.KeypointParams(format="xyas", remove_invisible=False),
)


def get_transforms_for_kp_regression(image_size: int = 64):
    return A.Compose(
        [
            A.HorizontalFlip(),
            # A.Rotate(limit=30, p=0.8),
            A.ShiftScaleRotate(
                shift_limit=0.25,
                scale_limit=(-0.15, 0.5),
                rotate_limit=30,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.5,
            ),
            # A.RandomSizedCrop(
            #     (
            #         int(image_size * 0.75),
            #         image_size,
            #     ),
            #     image_size,
            #     image_size,
            #     p=0.5,
            # ),
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
            # A.MotionBlur(),
        ],
        # format="xyas" is required while using tf.Data pipielines, otherwise
        # tf cannot validate the output shapes # TODO check if this is correct
        # remove_invisible=False is required to preserve the order and number of keypoints
        keypoint_params=A.KeypointParams(format="xyas", remove_invisible=False),
    )


def get_transforms_for_segmentation_masks(image_size: int = 64):
    return A.Compose(
        [
            # A.HorizontalFlip(),
            # A.VerticalFlip(),
            A.RandomRotate90(),
            A.RandomSizedCrop(
                (
                    image_size // 2,
                    image_size,
                ),
                image_size,
                image_size,
                # interpolation=cv2.INTER_CUBIC,
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
            A.Resize(
                image_size,
                image_size,
                # interpolation=cv2.INTER_CUBIC,
            ),  # always resize to desired shape
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


def kp_aug_fn(image, keypoints, image_size):
    data = {
        "image": image,
        "keypoints": keypoints,
    }
    aug_data = get_transforms_for_kp_regression(image_size)(**data)
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


def encode_kp_coordinates(image, kps):
    # TODO check if X and Y are in correct places
    # center
    kps_c_x = kps[0, 0] / image.shape[0]
    kps_c_y = kps[0, 1] / image.shape[1]
    # top
    kps_t_x = kps[1, 0] / image.shape[0]
    kps_t_y = kps[1, 1] / image.shape[1]
    kps = tf.stack((kps_c_x, kps_c_y, kps_t_x))
    return image, kps


def mask_aug_fn(image, mask, image_size):
    data = {
        "image": image,
        "mask": mask,
    }
    aug_data = get_transforms_for_segmentation_masks(image_size)(**data)
    aug_img = aug_data["image"]
    aug_mask = aug_data["mask"]
    aug_mask = tf.cast(aug_mask, tf.float32)
    return aug_img, aug_mask


def resize_fn(image, mask, image_size):
    data = {
        "image": image,
        "mask": mask,
    }
    aug_data = A.Resize(
        image_size,
        image_size,
        # interpolation=cv2.INTER_CUBIC,
    )(**data)
    aug_img = aug_data["image"]
    aug_mask = aug_data["mask"]
    aug_mask = tf.cast(aug_mask, tf.float32)
    return aug_img, aug_mask


def augment_kp_data(image, kp, image_size):
    image, kp = tf.numpy_function(
        func=kp_aug_fn,
        inp=[image, kp, image_size],
        Tout=(tf.uint8, tf.float32),
    )
    return image, kp


def add_noise_to_kp_coordinates(image, kp, sigma=0.01):

    kp = tf.random.normal(tf.shape(kp), stddev=sigma) + kp
    return image, kp


def augment_mask_data(image, mask, image_size):
    image, mask = tf.numpy_function(
        func=mask_aug_fn,
        inp=[image, mask, image_size],
        Tout=(tf.uint8, tf.float32),
    )
    return image, mask


def resize_mask_data(image, mask, image_size):
    image, mask = tf.numpy_function(
        func=resize_fn,
        inp=[image, mask, image_size],
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


def view_images_and_segmentation_prediction(ds, model):

    batch = next(iter(ds))  # extract 1 batch from the dataset
    image, masks = batch[0], batch[1]
    preds = model.predict(image)
    print(preds.shape)
    image = image.numpy()
    masks = masks.numpy()

    fig, axarr = plt.subplots(5, 4, figsize=(15, 15))
    for i in range(5):
        ax = axarr[i]
        img = image[i]
        ax_idx = 0

        mask = (masks[i, :, :, -1] * 255).astype("uint8")
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        ax[ax_idx].imshow(mask)
        ax[ax_idx].set_title("Masks")

        ax_idx += 1
        pred = (preds[i, :, :, -1] * 255).astype("uint8")
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
        ax[ax_idx].imshow(pred)
        ax[ax_idx].set_title("Pred")

        ax_idx += 1
        merged = cv2.addWeighted(img, 0.5, mask, 1.0, 0.0)
        ax[ax_idx].imshow(merged)
        ax[ax_idx].set_title("Masks + Image")
        ax_idx += 1

        pred_merged = cv2.addWeighted(img, 0.5, pred, 1.0, 0.0)
        ax[ax_idx].imshow(pred_merged)
        ax[ax_idx].set_title("Pred + Image")


def view_image_and_segmasks(ds):

    batch = next(iter(ds))  # extract 1 batch from the dataset
    image, masks = batch[0], batch[1]
    image = image.numpy()
    masks = masks.numpy()

    if len(masks.shape) > 2 and masks.shape[-1] > 1:
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
            mask = (masks[i, :, :, -1] * 255).astype("uint8")
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            ax[ax_idx].imshow(mask)
            ax[ax_idx].set_title("Masks")
            merged = cv2.addWeighted(img, 0.5, mask, 1.0, 0.0)

            ax_idx += 1
            ax[ax_idx].imshow(merged)
            ax[ax_idx].set_title("Masks + Image")


def view_image_and_kp(ds, max_examples: int = 5):

    batch = next(iter(ds))  # extract 1 batch from the dataset
    image, kps = batch[0], batch[1]
    image = image.numpy()
    # fake y coordinate for top
    kps = kps.numpy().reshape(-1, 3)
    zero_column = np.zeros(len(kps)).reshape(-1, 1) + 0.25
    kps = np.hstack((kps, zero_column))
    kps = kps.reshape(-1, 2, 2)

    kps[:, :, 0] *= image[0].shape[0]
    kps[:, :, 1] *= image[0].shape[1]
    n_examples = min(len(kps), max_examples)
    fig, axarr = plt.subplots(n_examples, 1, figsize=(15, 15))
    for i in range(n_examples):
        ax = axarr[i]
        img = image[i]
        kp = kps[i]
        ax.imshow(img.astype("uint8"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Image")
        ax.scatter(*kp[0, :2], marker="x")
        ax.scatter(*kp[1, :2], marker="x")


def view_images_and_kp_prediction(ds, model, n_kp=2, max_examples: int = 5):

    batch = next(iter(ds))  # extract 1 batch from the dataset
    image, kps = batch[0], batch[1]
    preds = model.predict(image)

    zero_column = np.zeros(len(preds)).reshape(-1, 1) + 0.25
    preds = np.hstack((preds, zero_column))
    preds = preds.reshape(-1, n_kp, 2)
    image = image.numpy()

    kps = kps.numpy().reshape(-1, 3)
    kps = np.hstack((kps, zero_column))
    kps = kps.reshape(-1, n_kp, 2)

    kps[:, :, 0] *= image[0].shape[0]
    kps[:, :, 1] *= image[0].shape[1]

    preds[:, :, 0] *= image[0].shape[0]
    preds[:, :, 1] *= image[0].shape[1]
    n_examples = min(len(kps), max_examples)
    fig, axarr = plt.subplots(n_examples, 2, figsize=(15, 15))
    for i in range(n_examples):
        ax = axarr[i][0]
        img = image[i]
        kp = kps[i]
        ax.imshow(img.astype("uint8"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Image")
        ax.scatter(*kp[0, :2])
        ax.scatter(*kp[1, :2])

        ax = axarr[i][1]
        pred_kp = preds[i]

        ax.imshow(img.astype("uint8"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Image")
        ax.scatter(*pred_kp[0, :2])
        ax.scatter(*pred_kp[1, :2])


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
        augmentation_fn = partial(augment_mask_data, image_size=image_size[0])
        dataset = dataset.map(
            augmentation_fn,
            num_parallel_calls=AUTOTUNE,
        )
    else:
        resize_fn = partial(resize_mask_data, image_size=image_size[0])
        dataset = dataset.map(
            resize_fn,
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


def get_watch_keypoints_heatmap_dataset(
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


def get_watch_keypoints_coordinates_dataset(
    X,
    y,
    augment: bool = True,
    noise: bool = False,
    batch_size=32,
    image_size=None,
    shuffle=True,
) -> tf.data.Dataset:
    set_shape_f = partial(set_shapes, img_shape=(*image_size, 3), target_shape=(2, 4))
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if augment:
        augmentation_fn = partial(augment_kp_data, image_size=image_size[0])
        dataset = dataset.map(
            augmentation_fn,
            num_parallel_calls=AUTOTUNE,
        )
    if noise:
        noise_fn = partial(add_noise_to_kp_coordinates, sigma=2)
        dataset = dataset.map(
            noise_fn,
            num_parallel_calls=AUTOTUNE,
        )
    dataset = dataset.map(set_shape_f, num_parallel_calls=AUTOTUNE).map(
        encode_kp_coordinates, num_parallel_calls=AUTOTUNE
    )
    if shuffle:
        dataset = dataset.shuffle(8 * batch_size)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
