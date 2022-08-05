from pathlib import Path

import numpy as np
from tensorflow import keras

from watch_recognition.train.object_detection_task import load_label_studio_dataset


def main():
    model = keras.models.load_model("models/detector/")
    model.summary()

    dataset_path = Path("datasets/watch-faces-local.json")
    train_data = list(
        load_label_studio_dataset(
            dataset_path,
            image_size=(512, 512),
            label_mapping={"WatchFace": 1},
            split="train",
        )
    )

    val_data = list(
        load_label_studio_dataset(
            dataset_path,
            image_size=(512, 512),
            label_mapping={"WatchFace": 1},
            split="val",
        )
    )
    print(len(train_data), len(val_data))

    for image, target_bboxes, target_classes in train_data:
        predictions = model.predict(np.expand_dims(image, 0))
        print(predictions)


if __name__ == "__main__":
    main()
