"""Test an uploaded model to Vertex AI."""

import os
import logging
import tensorflow as tf
import tensorflow_model_analysis as tfma

test_instance = {
    "feature_1": [1],
    "feature_2": [0],
    "feature_3": [1],
    "feature_4": [0],
    "feature_5": [1],
}

SERVING_DEFAULT_SIGNATURE_NAME = "serving_default"

from google.cloud import aiplatform as vertex_ai


def test_model_artifact():
    
    feature_types = {
        "feature_1": tf.dtypes.int64,
        "feature_2": tf.dtypes.int64,
        "feature_3": tf.dtypes.int64,
        "feature_4": tf.dtypes.int64,
        "feature_5": tf.dtypes.int64,
    }

    new_test_instance = dict()
    for key in test_instance:
        new_test_instance[key] = tf.constant(
            [test_instance[key]], dtype=feature_types[key]
        )

    print(new_test_instance)

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    model_display_name = os.getenv("MODEL_DISPLAY_NAME")

    assert project, "Environment variable PROJECT is None!"
    assert region, "Environment variable REGION is None!"
    assert model_display_name, "Environment variable MODEL_DISPLAY_NAME is None!"

    vertex_ai.init(project=project, location=region,)

    models = vertex_ai.Model.list(
        filter=f'display_name={model_display_name}',
        order_by="update_time"
    )

    assert (
        models
    ), f"No model with display name {model_display_name} exists!"

    model = models[-1]
    artifact_uri = model.gca_resource.artifact_uri
    print(f"Model artifact uri: {artifact_uri}")
    assert tf.io.gfile.exists(
        artifact_uri
    ), f"Model artifact uri {artifact_uri} does not exist!"

    saved_model = tf.saved_model.load(artifact_uri)
    logging.info("Model loaded successfully.")

    assert (
        SERVING_DEFAULT_SIGNATURE_NAME in saved_model.signatures
    ), f"{SERVING_DEFAULT_SIGNATURE_NAME} not in model signatures!"

    prediction_fn = saved_model.signatures["serving_default"]
    predictions = prediction_fn(**new_test_instance)
    logging.info("Model produced predictions.")

    print(predictions)
    assert predictions["outputs"].shape == (
        1,
        12,
    ), f"Invalid outputs shape: {predictions['outputs'].shape}!"


def test_model_endpoint():

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    model_display_name = os.getenv("MODEL_DISPLAY_NAME")
    endpoint_display_name = os.getenv("ENDPOINT_DISPLAY_NAME")

    assert project, "Environment variable PROJECT is None!"
    assert region, "Environment variable REGION is None!"
    assert model_display_name, "Environment variable MODEL_DISPLAY_NAME is None!"
    assert endpoint_display_name, "Environment variable ENDPOINT_DISPLAY_NAME is None!"

    endpoints = vertex_ai.Endpoint.list(
        filter=f'display_name={endpoint_display_name}',
        order_by="update_time"
    )
    assert (
        endpoints
    ), f"Endpoint with display name {endpoint_display_name} does not exist! in region {region}"

    endpoint = endpoints[-1]
    print(f"Calling endpoint: {endpoint}.")

    prediction = endpoint.predict([test_instance]).predictions[0]

    print(f"Prediction output: {prediction}")
    
    assert (
        len(prediction) > 0
    ), f"There must be some output returned: {len(prediction)}!"
    
