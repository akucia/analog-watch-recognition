import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.draw import rectangle, rectangle_perimeter

DEFAULT_TRANSFORMS = A.Compose(
    [
        A.RandomSizedCrop(min_max_height=(200, 224), height=224, width=224, p=0.5),
        A.ShiftScaleRotate(
            p=0.8,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(p=0.7),
                A.RGBShift(p=0.7),
                A.ChannelShuffle(p=0.7),
            ],
            p=1,
        ),
        A.Blur(blur_limit=5, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
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


def process_kp_data(image, kp):
    image, kp = tf.numpy_function(
        func=kp_aug_fn,
        inp=[image, kp],
        Tout=(tf.uint8, tf.float32),
    )
    return image, kp


def set_shapes(img, target, img_shape=(224, 224, 3), target_shape=(28, 28, 4)):
    img.set_shape(img_shape)
    target.set_shape(target_shape)
    return img, target


def view_image(ds):
    image, masks, _ = next(iter(ds))  # extract 1 batch from the dataset
    image = image.numpy()
    masks = masks.numpy()
    fig, axarr = plt.subplots(5, 2, figsize=(15, 15))
    for i in range(5):
        ax = axarr[i]
        img = image[i]
        ax_idx = 0
        ax[ax_idx].imshow(img)
        ax[ax_idx].imshow(img.astype("uint8"))
        ax[ax_idx].set_xticks([])
        ax[ax_idx].set_yticks([])
        ax[ax_idx].set_title("Image")

        ax_idx += 1
        ax[ax_idx].imshow(masks[i, :, :, -1].astype("uint8"))
        ax[ax_idx].set_title("Masks")


def encode_keypoints_to_mask_np(
    keypoints,
    image_size,
    mask_size,
    extent=(2, 2),
    include_background=True,
    separate_hour_and_minute_hands: bool = True,
    add_perimeter: bool = False,
):
    downsample_factor = image_size[0] / mask_size[0]
    all_masks = []
    points = keypoints[:, :2]
    fm_point = points / downsample_factor
    int_points = np.floor(fm_point).astype(int)
    # center and top
    for int_point in int_points[:2]:
        mask = _encode_point_to_mask(extent, int_point, mask_size)
        all_masks.append(mask)
    # hour and minute hands
    if separate_hour_and_minute_hands:
        for int_point in int_points[2:]:
            mask = _encode_point_to_mask(extent, int_point, mask_size)
            all_masks.append(mask)
    else:
        mask = np.zeros(mask_size, dtype=np.float32)
        for int_point in int_points[2:]:
            if extent[0] > 1:
                coords = tuple(int_point - 1)
            else:
                coords = tuple(int_point)
            rr, cc = rectangle(coords, extent=extent, shape=mask.shape)
            mask[cc, rr] = 1
            if add_perimeter:
                rr, cc = rectangle_perimeter(coords, extent=extent, shape=mask.shape)
                mask[cc, rr] = 0.5
        all_masks.append(mask)

    masks = np.array(all_masks).transpose((1, 2, 0))
    # background mask
    if include_background:
        background_mask = ((np.ones(mask_size) - masks.sum(axis=-1)) > 0).astype(
            "float32"
        )
        background_mask = np.expand_dims(background_mask, axis=-1)
        masks = np.concatenate((masks, background_mask), axis=-1)
    return np.expand_dims(np.argmax(masks, axis=-1).astype("float32"), axis=-1)


def _encode_point_to_mask(extent, int_point, mask_size):
    matched = np.zeros(mask_size, dtype=np.float32)
    if extent[0] > 1:
        coords = tuple(int_point - 1)
    else:
        coords = tuple(int_point)
    rr, cc = rectangle(coords, extent=extent, shape=matched.shape)
    matched[cc, rr] = 1
    return matched


def encode_keypoints_to_mask(
    image,
    keypoints,
    image_size,
    mask_size,
    extent,
    include_background=True,
    separate_hour_and_minute_handles=False,
):
    mask = tf.numpy_function(
        func=encode_keypoints_to_mask_np,
        inp=[
            keypoints,
            image_size,
            mask_size,
            extent,
            include_background,
            separate_hour_and_minute_handles,
        ],
        Tout=tf.float32,
    )
    return image, mask
