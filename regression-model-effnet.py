#!/usr/bin/env python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from watch_recognition.data_preprocessing import (
    load_data,
    load_synthethic_data,
    preprocess_targets,
    unison_shuffled_copies,
)
from watch_recognition.models import get_model
from watch_recognition.reports import (
    generate_report,
    plot_confusion_matrix,
    plot_to_image,
    predict_on_image,
)

plt.style.use("ggplot")

IMAGE_SIZE = (224, 224)

MODEL_NAME = "effnetb0"
TYPE = "classification"
RUNS_FILE = Path("runs.json")
if not RUNS_FILE.exists():
    RUN = 0
    data = {MODEL_NAME: RUN}
else:
    with RUNS_FILE.open("r") as f:
        data = json.load(f)
    if MODEL_NAME in data:
        data[MODEL_NAME] += 1
    else:
        data[MODEL_NAME] = 0
    RUN = data[MODEL_NAME]
with RUNS_FILE.open("w") as f:
    json.dump(data, f)

print(f"using model {MODEL_NAME} run: {RUN}")


def main():
    model = get_model(IMAGE_SIZE, backbone=MODEL_NAME, kind=TYPE)
    model.summary()
    synth = "./data/analog_clocks/label.csv"

    X_train_synth, y_train_synth = load_synthethic_data(
        Path(synth), IMAGE_SIZE, n_samples=500
    )

    X_train, y_train = load_data(
        Path("./data/watch-time-train/labels.csv"),
        IMAGE_SIZE,
    )
    X_val, y_val = load_data(
        Path("./data/watch-time-validation/labels.csv"),
        IMAGE_SIZE,
    )

    X_train = np.vstack((X_train, X_train_synth))
    y_train = pd.concat((y_train, y_train_synth))
    X_train, y_train = unison_shuffled_copies(X_train, y_train)

    print(len(y_train), len(X_train))

    print(len(y_val), len(X_val))

    y_train = preprocess_targets(y_train, kind=TYPE)
    y_val = preprocess_targets(y_val, kind=TYPE)

    # plt.hist(y_train["hour"], bins=12)

    EPOCHS = 1000
    logdir = f"tensorboard_logs/{TYPE}/{MODEL_NAME}/run_{RUN}"

    file_writer_cm = tf.summary.create_file_writer(logdir + "/cm")

    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = model.predict(X_val)["hour"]
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        test_target = np.argmax(y_val["hour"], axis=1)
        cm = confusion_matrix(test_target, test_pred)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(
            cm, list(map(str, [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        )
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix Hours (val)", cm_image, step=epoch)

    def log_confusion_matrix_train(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = model.predict(X_train)["hour"]
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        test_target = np.argmax(y_train["hour"], axis=1)
        cm = confusion_matrix(test_target, test_pred)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(
            cm, list(map(str, [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        )
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix Hours (train)", cm_image, step=epoch)

    # Define the per-epoch callback.
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    cm_callback_train = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=log_confusion_matrix_train
    )
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=logdir,
                update_freq="epoch",
            ),
            cm_callback,
            cm_callback_train,
        ],
    )

    model.save(f"./models/{TYPE}/{MODEL_NAME}/run_{RUN}")

    # H = model.history
    # lossNames = [hi for hi in H.history.keys() if "val" not in hi]
    #
    # (fig, ax) = plt.subplots(len(lossNames), 1, figsize=(16, 9))
    # # loop over the loss names
    # for (i, l) in enumerate(lossNames):
    #     # plot the loss for both the training and validation data
    #     title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    #     ax[i].set_title(title)
    #     ax[i].set_xlabel("Epoch #")
    #     ax[i].set_ylabel("Loss")
    #     n_steps = len(H.history[l])
    #     ax[i].plot(np.arange(0, n_steps), H.history[l], label=l)
    #     ax[i].plot(np.arange(0, n_steps), H.history["val_" + l], label="val_" + l)
    #     ax[i].legend()
    _ = generate_report(X_train, y_train, model, kind=TYPE)
    _ = generate_report(X_val, y_val, model, kind=TYPE)

    predict_on_image("example_data/test-image-2.jpg", model, image_size=IMAGE_SIZE)


if __name__ == "__main__":
    main()
