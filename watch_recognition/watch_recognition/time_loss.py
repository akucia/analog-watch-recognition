import numpy as np
import tensorflow as tf


def time_loss_np(yt: np.ndarray, yp: np.ndarray, max_val: int = 12) -> np.ndarray:
    abs_dist = np.abs(yt - yp)
    abs_dist_2 = np.abs(max_val - np.maximum(yt, yp)) + np.minimum(yt, yp)
    return np.where(abs_dist > max_val / 2, abs_dist_2, abs_dist)


def time_loss_tf(yt: tf.Tensor, yp: tf.Tensor, max_val: float = 12.0) -> tf.Tensor:
    abs_dist = tf.abs(yt - yp)
    abs_dist_2 = tf.abs(max_val - tf.maximum(yt, yp)) + tf.minimum(yt, yp)
    return tf.where(abs_dist > max_val / 2.0, abs_dist_2, abs_dist)
