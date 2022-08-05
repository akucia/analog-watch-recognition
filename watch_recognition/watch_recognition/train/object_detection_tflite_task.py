import tensorflow as tf

assert tf.__version__.startswith("2")

tf.get_logger().setLevel("ERROR")
from absl import logging

logging.set_verbosity(logging.ERROR)


def main():
    pass
    train_data, validation_data, test_data = object_detector.DataLoader.from_csv(
        "gs://cloud-ml-data/img/openimage/csv/salads_ml_use.csv"
    )
    spec = model_spec.get("efficientdet_lite0")


if __name__ == "__main__":
    main()
