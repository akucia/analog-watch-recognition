import functools

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model

from watch_recognition.time_loss import time_loss_tf


def hour_diff(y_true, y_pred):
    diff = tf.round(tf.abs(y_true - y_pred) * 12)
    return tf.reduce_mean(diff, axis=-1)  # Note the `axis=-1`


def minutes_diff(y_true, y_pred):
    diff = tf.round(tf.abs(y_true - y_pred) * 60)
    return tf.reduce_mean(diff, axis=-1)  # Note the `axis=-1


class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale, *args, **kwargs):
        super(ScaleLayer, self).__init__(*args, **kwargs)
        self.scale = tf.Variable(float(scale))

    def call(self, inputs):
        return inputs * self.scale


def get_model(image_size, dropout=0.5, backbone="custom", kind="regression"):
    IMAGE_SIZE = image_size
    inp = Input(shape=(*IMAGE_SIZE, 3))
    conv2d_layer = functools.partial(
        tf.keras.layers.SeparableConv2D,
        depth_multiplier=1,
        # pointwise_initializer=tf.initializers.variance_scaling(),
        # depthwise_initializer=tf.initializers.variance_scaling(),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    relu = tf.nn.relu6

    if backbone == "custom":
        # Convolutional Layers
        x = conv2d_layer(100, kernel_size=3, activation=relu, padding="same")(inp)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = conv2d_layer(100, kernel_size=3, activation=relu, padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = conv2d_layer(100, kernel_size=3, activation=relu, padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = conv2d_layer(100, kernel_size=3, activation=relu, padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = conv2d_layer(150, kernel_size=3, activation=relu, padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = conv2d_layer(200, kernel_size=3, activation=relu, padding="same")(x)
        x = Dropout(dropout)(x)
    elif backbone == "effnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
        )
        # base_model.trainable = False
        x = base_model(inp)

    # Hour branch
    # hour = conv2d_layer(100, kernel_size=3, padding="same", activation=relu)(x)
    # hour = MaxPooling2D(pool_size=(2, 2))(hour)
    # x = tf.keras.layers.BatchNormalization()(x)
    hour = Flatten()(x)
    hour = Dense(100, activation=relu)(hour)
    x = Dropout(dropout)(x)

    # Minute Branch
    minute = conv2d_layer(100, kernel_size=3, padding="same", activation=relu)(x)
    # minute = MaxPooling2D(pool_size=(2, 2))(minute)
    # x = tf.keras.layers.BatchNormalization()(x)
    minute = Flatten()(minute)
    minute = Dense(100, activation=relu)(minute)

    if kind == "regression":
        hour = Dense(1, activation="tanh")(hour)
        hour = ScaleLayer(2, name="hour")(hour)

        minute = Dense(1, activation="tanh", name="minute")(minute)
        # minute = ScaleLayer()(minute)
        time_loss = functools.partial(time_loss_tf, max_val=1)
        losses = {
            "hour": "mae",
            # "minute": time_loss,
        }
    elif kind == "classification":
        hour = Dense(
            12,
            name="hour",
            bias_initializer="random_normal",
        )(hour)
        minute = Dense(60, name="minute")(minute)
        losses = {
            "hour": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            # "minute": time_loss,
        }
    else:
        raise ValueError(f"unrecognized kind {kind}")

    # loss_weights = {"hour": 1.0, "minute": 0.0}
    loss_weights = {"hour": 1.0}

    # model = Model(inputs=inp, outputs={"hour": hour, "minute": minute})
    model = Model(inputs=inp, outputs={"hour": hour})

    # metrics = {"hour": ["mae"], "minute": ["mae"]}
    # metrics = {"hour": ["mae"]}
    metrics = None
    model.compile(
        loss=losses, optimizer="adam", metrics=metrics, loss_weights=loss_weights
    )
    return model


def export_tflite(model, export_path, quantize: bool = False, test_image=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("quantaizing model")

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


def build_backbone(image_size):
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(*image_size, 3),
        include_top=False,
    )
    # outputs = [
    #     base_model.get_layer(layer_name).output
    #     for layer_name in ["block3b_project_conv"]
    # ]
    outputs = [
        base_model.get_layer(layer_name).output
        for layer_name in ["block5c_project_conv"]
    ]
    return tf.keras.Model(inputs=[base_model.inputs], outputs=outputs)


def decode_batch_predictions(predicted):
    output_2d_shape = predicted.shape[1:3]
    minutes = []
    hours = []
    for row in range(predicted.shape[0]):
        # TODO take argmax for all outputs at once instead of for loop
        # center
        center = np.array(
            np.unravel_index(np.argmax(predicted[row, :, :, 0]), output_2d_shape)
        )[::-1]
        # top
        top = (
            np.array(
                np.unravel_index(np.argmax(predicted[row, :, :, 1]), output_2d_shape)
            )[::-1]
            - center
        )

        # hour
        hour = (
            np.array(
                np.unravel_index(np.argmax(predicted[row, :, :, 2]), output_2d_shape)
            )[::-1]
            - center
        )
        # minute
        minute = (
            np.array(
                np.unravel_index(np.argmax(predicted[row, :, :, 3]), output_2d_shape)
            )[::-1]
            - center
        )

        read_hour = (
            np.rad2deg(np.arctan2(top[0], top[1]) - np.arctan2(hour[0], hour[1]))
            / 360
            * 12
        )
        read_minute = (
            np.rad2deg(np.arctan2(top[0], top[1]) - np.arctan2(minute[0], minute[1]))
            / 360
            * 60
        )
        hours.append(read_hour)
        minutes.append(read_minute)
    read_hour = np.array(hours)
    read_hour = np.floor(read_hour.reshape(-1, 1)).astype(int)
    read_minute = np.array(minutes)
    read_minute = np.round(read_minute.reshape(-1, 1)).astype(int)
    read_hour = read_hour % 12
    read_hour = np.where(read_hour == 0, 12, read_hour)
    read_minute = read_minute % 60

    return np.hstack((read_hour, read_minute))


def decode_predictions(predicted):
    center = np.array(
        np.unravel_index(np.argmax(predicted[0, :, :, 0]), predicted.shape[1:3])
    )[::-1]
    hour = (
        np.array(
            np.unravel_index(np.argmax(predicted[0, :, :, 1]), predicted.shape[1:3])
        )[::-1]
        - center
    )
    minute = (
        np.array(
            np.unravel_index(np.argmax(predicted[0, :, :, 2]), predicted.shape[1:3])
        )[::-1]
        - center
    )
    top = (
        np.array(
            np.unravel_index(np.argmax(predicted[0, :, :, 3]), predicted.shape[1:3])
        )[::-1]
        - center
    )
    read_hour = (
        np.rad2deg(np.arctan2(top[0], top[1]) - np.arctan2(hour[0], hour[1])) / 360 * 12
    )
    if read_hour < 0:
        read_hour += 12
    read_minute = (
        np.rad2deg(np.arctan2(top[0], top[1]) - np.arctan2(minute[0], minute[1]))
        / 360
        * 60
    )
    if read_minute < 0:
        read_minute += 60
    return read_hour, read_minute


def custom_focal_loss(target, output, gamma=4, alpha=0.25):
    epsilon_ = tf.keras.backend.epsilon()
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    bce = tf.pow(target, gamma) * tf.math.log(output + epsilon_) * (1 - alpha)
    bce += tf.pow(1 - target, gamma) * tf.math.log(1 - output + epsilon_) * alpha
    return -bce


class Involution(tf.keras.layers.Layer):
    """https://keras.io/examples/vision/involution/"""

    def __init__(
        self, channel, group_number, kernel_size, stride, reduction_ratio, name
    ):
        super().__init__(name=name)

        # Initialize the parameters.
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Get the shape of the input.
        (_, height, width, num_channels) = input_shape

        # Scale the height and width with respect to the strides.
        height = height // self.stride
        width = width // self.stride

        # Define a layer that average pools the input tensor
        # if stride is more than 1.
        self.stride_layer = (
            tf.keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )
        # Define the kernel generation layer.
        self.kernel_gen = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        # Define reshape layers
        self.kernel_reshape = tf.keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = tf.keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = tf.keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )

    def call(self, x):
        # Generate the kernel with respect to the input tensor.
        # B, H, W, K*K*G
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)

        # reshape the kerenl
        # B, H, W, K*K, 1, G
        kernel = self.kernel_reshape(kernel)

        # Extract input patches.
        # B, H, W, K*K*C
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape the input patches to align with later operations.
        # B, H, W, K*K, C//G, G
        input_patches = self.input_patches_reshape(input_patches)

        # Compute the multiply-add operation of kernels and patches.
        # B, H, W, K*K, C//G, G
        output = tf.multiply(kernel, input_patches)
        # B, H, W, C//G, G
        output = tf.reduce_sum(output, axis=3)

        # Reshape the output kernel.
        # B, H, W, C
        output = self.output_reshape(output)

        # Return the output tensor and the kernel.
        return output, kernel
