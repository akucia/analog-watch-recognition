from typing import List, Optional

import cv2
import numpy as np
import tensorflow as tf
from distinctipy import distinctipy
from matplotlib import pyplot as plt

from watch_recognition.utilities import BBox, Point


def visualize_keypoints(image: np.ndarray, points: List[Point], savefile=None):
    plt.figure()
    plt.tight_layout()
    plt.axis("off")
    plt.imshow(image)
    colors = ["red", "green", "blue"]
    ax = plt.gca()
    for point, color in zip(points, colors):
        point.plot(color=color, size=30, ax=ax)
    plt.legend()
    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")
    return ax


def visualize_masks(image: np.ndarray, masks: List[np.ndarray], savefile=None, ax=None):
    ax = ax or plt.gca()
    overlay = draw_masks(image, masks)
    ax.imshow(overlay)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")
    return ax


def draw_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    colors: Optional[List[np.ndarray]] = None,
):
    colors = colors or distinctipy.get_colors(len(masks))
    overlay = image.astype("uint8")

    for mask, color in zip(masks, colors):
        heatmap = np.zeros(shape=(*mask.shape[:2], 3)).astype("uint8")
        heatmap[mask] = (np.array(color) * 255).astype("uint8")
        overlay = cv2.addWeighted(overlay, 1.0, heatmap, 0.5, 0.0)
    return overlay


def visualize_detections(
    image, boxes, classes, scores, figsize=(10, 10), savefile=None
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        bbox = BBox.from_ltwh(*box, name=text, score=score)
        bbox.plot(ax=ax)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")
    return ax


def visualize_segmentation_dataset(dataset: tf.data.Dataset):
    example = next(iter(dataset))
    images, all_masks = example["images"], example["segmentation_masks"]
    images, all_masks = images.numpy(), all_masks.numpy()
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    max_imgs = min(9, len(images))
    for i in range(max_imgs):
        ax = plt.subplot(9 // 3, 9 // 3, i + 1)
        ax.set_axis_off()
        image = images[i]
        masks = all_masks[i].squeeze() > 0
        masks = np.split(masks, masks.shape[-1], axis=-1)
        masks = [mask.squeeze() for mask in masks]
        visualize_masks(image, masks, ax=ax)
