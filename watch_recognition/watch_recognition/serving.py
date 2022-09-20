import tensorflow as tf
from tensorflow_serving.apis import model_pb2, predict_pb2, prediction_log_pb2


def save_tf_serving_warmup_request(example_image_np, warmup_tf_record_file):
    warmup_tf_record_file.parent.mkdir(exist_ok=True, parents=True)
    with tf.io.TFRecordWriter(str(warmup_tf_record_file)) as writer:
        tensor_proto = tf.make_tensor_proto(example_image_np)
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(signature_name="serving_default"),
            inputs={"image": tensor_proto},
        )
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request)
        )
        writer.write(log.SerializeToString())
