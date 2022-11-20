from typing import Tuple

import numpy as np
import segmentation_models as sm
import tensorflow as tf
from matplotlib import pyplot as plt
from segmentation_models.base import Loss
from segmentation_models.base.functional import average, get_reduce_axes
from segmentation_models.losses import SMOOTH

from watch_recognition.utilities import Line, Point

sm.set_framework("tf.keras")


def hour_diff(y_true, y_pred):
    diff = tf.round(tf.abs(y_true - y_pred) * 12)
    return tf.reduce_mean(diff, axis=-1)


def minutes_diff(y_true, y_pred):
    diff = tf.round(tf.abs(y_true - y_pred) * 60)
    return tf.reduce_mean(diff, axis=-1)


def export_tflite(model, export_path, quantize: bool = False, test_image=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("quantaizing model")

        # TODO allow to use more images
        def representative_dataset():
            for data in (
                tf.data.Dataset.from_tensor_slices(test_image).batch(1).take(100)
            ):
                yield [tf.dtypes.cast(data, tf.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
    else:
        optimizations = [tf.lite.Optimize.DEFAULT]
        converter.optimizations = optimizations

    tflite_model = converter.convert()

    with tf.io.gfile.GFile(export_path, "wb") as f:
        f.write(tflite_model)
    print(f"model exported to {export_path}")


def build_backbone(image_size, backbone_layer="block5c_project_conv"):
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        input_shape=(*image_size, 3),
        include_top=False,
    )
    outputs = [
        base_model.get_layer(layer_name).output for layer_name in [backbone_layer]
    ]
    return tf.keras.Model(inputs=[base_model.inputs], outputs=outputs)


# class IoULoss(nn.Module):
#     """
#     Intersection over Union Loss.
#     IoU = Area of Overlap / Area of Union
#     IoU loss is modified to use for heatmaps.
#     """
#
#     def __init__(self):
#         super(IoULoss, self).__init__()
#         self.EPSILON = 1e-6
#
#     def _op_sum(self, x):
#         return x.sum(-1).sum(-1)
#
#     def forward(self, y_pred, y_true):
#         inter = self._op_sum(y_true * y_pred)
#         union = (
#             self._op_sum(y_true ** 2)
#             + self._op_sum(y_pred ** 2)
#             - self._op_sum(y_true * y_pred)
#         )
#         iou = (inter + self.EPSILON) / (union + self.EPSILON)
#         iou = torch.mean(iou)
#         return 1 - iou
#
#
class IouLoss2(Loss):
    """
    Inspired by
    https://github.com/OlgaChernytska/2D-Hand-Pose-Estimation-RGB/blob/c9f201ca114129fa750f4bac2adf0f87c08533eb/utils/prep_utils.py#L88
    """

    def __init__(
        self, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH
    ):
        super().__init__(name="IOULoss")
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def iou_score(
        self,
        gt,
        pr,
        class_weights=1.0,
        smooth=SMOOTH,
        per_image=False,
        **kwargs,
    ):

        backend = kwargs["backend"]
        # gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
        # pr = round_if_needed(pr, threshold, **kwargs)
        axes = get_reduce_axes(per_image, **kwargs)

        # score calculation
        intersection = backend.sum(gt * pr, axis=axes)
        union = (
            backend.sum(gt * gt, axis=axes)
            + backend.sum(pr * pr, axis=axes)
            - backend.sum(gt * pr, axis=axes)
        )
        score = (intersection + smooth) / (union + smooth)
        score = average(score, per_image, class_weights, **kwargs)

        return score

    def __call__(self, gt, pr):
        return 1 - self.iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
            **self.submodules,
        )


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    https://stackoverflow.com/a/13849249
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def points_to_time(
    center: Point, hour: Point, minute: Point, top: Point, debug: bool = False
) -> Tuple[float, float]:
    # TODO unittests

    assert hour.name == "Hour", hour.name
    assert minute.name == "Minute", minute.name

    if debug:
        fig, ax = plt.subplots(1, 1)
        top.plot(color="green")
        hour.plot(color="red")
        center.plot(color="k")
        minute.plot(color="orange")
        ax.invert_yaxis()
        ax.legend()

    up = center.translate(0, -10)
    line_up = Line(center, up)
    line_top = Line(center, top)
    rot_angle = np.rad2deg(line_up.angle_between(line_top))

    top = top.rotate_around_origin_point(center, rot_angle)
    minute = minute.rotate_around_origin_point(center, rot_angle)
    hour = hour.rotate_around_origin_point(center, rot_angle)

    hour = hour.translate(-center.x, -center.y)
    minute = minute.translate(-center.x, -center.y)
    top = top.translate(-center.x, -center.y)
    center = center.translate(-center.x, -center.y)
    if debug:
        fig, ax = plt.subplots(1, 1)
        top.plot(color="green")
        hour.plot(color="red")
        center.plot(color="k")
        minute.plot(color="orange")
        ax.invert_yaxis()
        ax.legend()
    hour = hour.as_array
    minute = minute.as_array
    top = top.as_array
    hour_deg = np.rad2deg(angle_between(top, hour))
    # TODO verify how to handle negative angles
    if hour[0] < top[0]:
        hour_deg = 360 - hour_deg

    read_hour = hour_deg / 360 * 12
    read_hour = np.floor(read_hour).astype(int)
    read_hour = read_hour % 12
    if read_hour == 0:
        read_hour = 12
    minute_deg = np.rad2deg(angle_between(top, minute))
    # TODO verify how to handle negative angles
    if minute[0] < top[0]:
        minute_deg = 360 - minute_deg
    read_minute = minute_deg / 360 * 60
    read_minute = np.floor(read_minute).astype(int)
    read_minute = read_minute % 60
    return read_hour, read_minute


def custom_focal_loss(target, output, gamma=4, alpha=0.25):
    epsilon_ = tf.keras.backend.epsilon()
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    bce = tf.pow(target, gamma) * tf.math.log(output + epsilon_) * (1 - alpha)
    bce += tf.pow(1 - target, gamma) * tf.math.log(1 - output + epsilon_) * alpha
    return -bce


def get_model(image_size: Tuple[int, int] = (224, 224)) -> tf.keras.Model:
    backbone = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        input_shape=(*image_size, 3),
        include_top=False,
    )
    outputs = [
        backbone.get_layer(layer_name).output for layer_name in ["block7a_project_conv"]
    ]
    base_model = tf.keras.Model(inputs=[backbone.inputs], outputs=outputs)
    for layer in backbone.layers:
        if "project_conv" in layer.name:
            print(layer.name, layer.output.shape)
    inputs = tf.keras.Input(
        shape=(*image_size, 3),
    )
    x = base_model(inputs)
    for i in range(2):
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, padding="same", activation=None
        )(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

    x = tf.keras.layers.UpSampling2D()(x)
    for i in range(2):
        x = tf.keras.layers.MaxPool2D(padding="same", strides=1)(x)
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, padding="same", activation=None
        )(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
    output = tf.keras.layers.Conv2D(
        filters=4, kernel_size=1, strides=1, padding="same", activation="softmax"
    )(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model


def get_segmentation_model(
    image_size: Tuple[int, int] = (224, 224),
    n_outputs: int = 4,
    backbone="efficientnetb0",
) -> tf.keras.Model:

    inputs = tf.keras.Input(
        shape=(*image_size, 3),
    )
    sm_model = sm.FPN(
        backbone,
        classes=n_outputs,
        activation="sigmoid",
        input_shape=(*image_size, 3),
    )
    outputs = sm_model(inputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# deeplab v3
# https://keras.io/examples/vision/deeplabv3_plus/#building-the-deeplabv3-model


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_size: int, num_classes):
    model_input = tf.keras.Input(shape=(image_size, image_size, 3))
    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = tf.keras.layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = tf.keras.layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    x = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model_output = tf.keras.layers.Activation("sigmoid")(x)
    return tf.keras.Model(inputs=model_input, outputs=model_output)
