import logging
import tensorflow as tf
import tensorflow_transform as tft

from src.common import features

from tensorflow import keras

from typing import List
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2


from src.model_training import data, model_tfdf


def train(
    fn_args=None,
    csv_data_dir=None
):

    if csv_data_dir:
        train_dataset = data.get_csv_dataset(csv_data_dir)
        eval_dataset = data.get_csv_dataset(csv_data_dir)
    else:
        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
        schema = tf_transform_output.transformed_metadata.schema
        
        train_dataset = _input_fn(
            fn_args.train_files,
            fn_args.data_accessor,
            schema,
            features.TARGET_FEATURE_NAME,
            batch_size=1)

        eval_dataset = _input_fn(
            fn_args.eval_files,
            fn_args.data_accessor,
            schema,
            features.TARGET_FEATURE_NAME,
            batch_size=1)

    decision_forest = model_tfdf.create_decision_forest()
    if fn_args and fn_args.base_model:
        try:
            decision_forest = keras.load_model(fn_args.base_model)
        except:
            pass

    logging.info("Model training started...")
    decision_forest.fit(
        train_dataset,
        validation_data=eval_dataset,
    )
    logging.info("Model training completed.")

    return decision_forest


def evaluate(model, eval_data_dir):
    
    logging.info("Model evaluation started...")
    eval_dataset = data.get_csv_dataset(
        eval_data_dir,
    )

    evaluation_metrics = model.evaluate(eval_dataset)
    logging.info("Model evaluation completed.")

    return evaluation_metrics


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              label: str,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: A schema proto of input data.
    label: Name of the label.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
        batch_size=batch_size, 
        label_key=label, 
        num_epochs=1, 
        shuffle=False),
      schema)