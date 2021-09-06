import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.draw import rectangle

# transforms = A.Compose(
#     [
#         A.ChannelShuffle(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.Blur(blur_limit=11, p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.Rotate(p=0.5),
#     ],
#     additional_targets={
#         "mask0": "mask",
#         "mask1": "mask",
#         "mask2": "mask",
#         "mask3": "mask",
#     },
# )

transforms = A.Compose(
    [
        A.RandomSizedCrop(min_max_height=(112, 224), height=224, width=224, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5),
        A.OneOf(
            [
                A.HueSaturationValue(p=0.5),
                A.RGBShift(p=0.7),
                A.ChannelShuffle(p=0.5),
            ],
            p=1,
        ),
        A.Blur(blur_limit=5, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ],
    # keypoint_params=A.KeypointParams(format='xy'),
)


def aug_fn(image, masks, mask_size=(28, 28), image_size=(224, 224)):
    data = {
        "image": image,
    }
    for i in range(4):
        mask = masks[:, :, i]

        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_AREA)
        data[f"mask{i}"] = mask
    aug_data = transforms(**data)
    aug_img = aug_data["image"]

    aug_masks = [
        cv2.resize(aug_data[f"mask{i}"], mask_size, interpolation=cv2.INTER_AREA)
        for i in range(4)
    ]

    aug_masks = np.stack(aug_masks, axis=-1)
    aug_masks = tf.cast(aug_masks, tf.float32)
    return aug_img, aug_masks


def process_mask_data(image, masks, mask_size=(28, 28), image_size=(224, 224)):
    image, masks = tf.numpy_function(
        func=aug_fn,
        inp=[image, masks, mask_size, image_size],
        Tout=(tf.float32, tf.float32),
    )
    return image, masks


def kp_aug_fn(image, keypoints):
    data = {
        "image": image,
        "keypoints": keypoints,
    }
    aug_data = transforms(**data)
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
    # entry = next(iter(ds)) # extract 1 batch from the dataset
    image, mask = next(iter(ds))
    image = image.numpy()
    mask = mask.numpy()

    #     fig = plt.figure(figsize=(22, 22))
    fig, axarr = plt.subplots(5, 6, figsize=(22, 22))
    for i in range(5):
        ax = axarr[i]
        img = image[i]
        ax[0].imshow(img)
        ax[0].imshow(img.astype("uint8"))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Image")
        for j, tag in enumerate(["Center", "Top", "Hour", "Minute"]):
            ax_idx = j + 1
            ax[ax_idx].imshow(mask[i, :, :, j].astype("uint8"))
            ax[ax_idx].set_xticks([])
            ax[ax_idx].set_yticks([])
            ax[ax_idx].set_title(f"Point: {tag}")
        ax[5].imshow(mask[i, :, :, j + 1].astype("uint8"))
        ax[5].set_title("Background")


def encode_keypoints_to_mask_np(
    keypoints, image_size, mask_size, extent=(2, 2), include_background=True
):
    downsample_factor = image_size[0] / mask_size[0]
    all_masks = []
    points = keypoints[:, :2]
    fm_point = points / downsample_factor
    int_points = np.floor(fm_point).astype(int)
    for int_point in int_points:
        matched = np.zeros(mask_size, dtype=np.float32)
        if extent[0] > 1:
            coords = tuple(int_point - 1)
        else:
            coords = tuple(int_point)
        rr, cc = rectangle(coords, extent=extent, shape=matched.shape)
        matched[cc, rr] = 1
        all_masks.append(matched)

    masks = np.array(all_masks).transpose((1, 2, 0))
    # TODO debug this
    if include_background:
        background_mask = ((np.ones(mask_size) - masks.sum(axis=-1)) > 0).astype(
            "float32"
        )
        background_mask = np.expand_dims(background_mask, axis=-1)
        masks = np.concatenate((masks, background_mask), axis=-1)
    return masks


def encode_keypoints_to_mask(image, keypoints, image_size, mask_size, extent):
    mask = tf.numpy_function(
        func=encode_keypoints_to_mask_np,
        inp=[keypoints, image_size, mask_size, extent],
        Tout=tf.float32,
    )
    return image, mask
