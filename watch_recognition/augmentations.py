import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

transforms = A.Compose(
    [
        A.ChannelShuffle(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Blur(blur_limit=11, p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(p=0.5),
    ],
    additional_targets={
        "mask0": "mask",
        "mask1": "mask",
        "mask2": "mask",
        "mask3": "mask",
    },
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
    if np.max(aug_img) < 2:
        aug_img = tf.cast(aug_img * 255.0, tf.float32)
    aug_masks = [
        cv2.resize(aug_data[f"mask{i}"], mask_size, interpolation=cv2.INTER_AREA)
        for i in range(4)
    ]
    # for mask in aug_masks:
    #     print(mask.shape)

    aug_masks = np.stack(aug_masks, axis=-1)
    aug_masks = tf.cast(aug_masks, tf.float32)
    return aug_img, aug_masks


def process_data(image, masks, mask_size=(28, 28), image_size=(224, 224)):
    image, masks = tf.numpy_function(
        func=aug_fn,
        inp=[image, masks, mask_size, image_size],
        Tout=(tf.float32, tf.float32),
    )
    return image, masks


def set_shapes(img, masks, img_shape=(224, 224, 3), masks_shape=(28, 28, 4)):
    img.set_shape(img_shape)
    masks.set_shape(masks_shape)
    return img, masks


def view_image(ds):
    # entry = next(iter(ds)) # extract 1 batch from the dataset
    image, mask = next(iter(ds))
    image = image.numpy()
    mask = mask.numpy()

    #     fig = plt.figure(figsize=(22, 22))
    fig, axarr = plt.subplots(5, 5, figsize=(22, 22))
    for i in range(5):
        ax = axarr[i]
        print(image[i].max())
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
