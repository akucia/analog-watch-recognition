from typing import List

import numpy as np
from matplotlib import pyplot as plt

from watch_recognition.utilities import Point


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
