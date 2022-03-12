import logging

import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_transform.tf_metadata import schema_utils

from src.common import features


def _get_serve_tf_examples_fn(trained_model, raw_feature_spec):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        for key in list(raw_feature_spec.keys()):
            if key not in features.FEATURE_NAMES:
                raw_feature_spec.pop(key)

        parsed_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)

        outputs = trained_model(parsed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def _get_serve_features_fn(trained_model):
    """Returns a function that accept a dictionary of features and applies TFT."""

    @tf.function
    def serve_features_fn(raw_features):
        """Returns the output to be used in the serving signature."""

        outputs = trained_model(raw_features)
        return {"outputs": outputs}

    return serve_features_fn


def export_serving_model(
    trained_model, serving_model_dir, raw_schema_location
):

    raw_schema = tfdv.load_schema_text(raw_schema_location)
    raw_feature_spec = schema_utils.schema_as_feature_spec(raw_schema).feature_spec

    features_input_signature = {
        feature_name: tf.TensorSpec(
            shape=(None, 1), dtype=spec.dtype, name=feature_name
        )
        for feature_name, spec in raw_feature_spec.items()
        if feature_name in features.FEATURE_NAMES
    }

    signatures = {
        "serving_default": _get_serve_features_fn(
            trained_model
        ).get_concrete_function(features_input_signature),
        "serving_tf_example": _get_serve_tf_examples_fn(
            trained_model, raw_feature_spec
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    logging.info("Model export started...")
    trained_model.save(serving_model_dir, signatures=signatures)
    logging.info("Model export completed.")
