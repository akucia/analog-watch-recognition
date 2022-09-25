from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import model_pb2, predict_pb2, prediction_log_pb2


def save_tf_serving_warmup_request(
    request_image_batch: np.ndarray, model_save_dir: Path, dtype: str = "float32"
):
    if len(request_image_batch.shape) != 4:
        raise ValueError(
            f"input batch should be 4D, got shape {request_image_batch.shape} "
            f"({len(request_image_batch.shape)}D)"
        )
    warmup_tf_record_file = (
        model_save_dir / "assets.extra" / "tf_serving_warmup_requests"
    )
    warmup_tf_record_file.parent.mkdir(exist_ok=True, parents=True)
    with tf.io.TFRecordWriter(str(warmup_tf_record_file)) as writer:
        tensor_proto = tf.make_tensor_proto(request_image_batch.astype(dtype=dtype))
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(signature_name="serving_default"),
            inputs={"image": tensor_proto},
        )
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request)
        )
        writer.write(log.SerializeToString())
