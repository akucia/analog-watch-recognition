import functools

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
