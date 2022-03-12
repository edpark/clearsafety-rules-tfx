import pytest
import sys
import os
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
import logging

from src.tfx_pipelines import config
from src.tfx_pipelines import training_pipeline

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

MLMD_SQLLITE = "mlmd.sqllite"


def test_e2e_pipeline():

    if not tf.io.gfile.exists("data/data.csv"):
        logging.info("data/data.csv doesn't exist - exiting")
        return

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    model_display_name = os.getenv("MODEL_DISPLAY_NAME")
    dataset_display_name = os.getenv("DATASET_DISPLAY_NAME")
    gcs_location = os.getenv("GCS_LOCATION")
    gcs_data_location = os.path.join(os.getenv("GCS_LOCATION"), "data", "data.csv")
    model_registry = os.getenv("MODEL_REGISTRY_URI")
    upload_model = os.getenv("UPLOAD_MODEL")

    assert project, "Environment variable PROJECT is None!"
    assert region, "Environment variable REGION is None!"
    assert dataset_display_name, "Environment variable DATASET_DISPLAY_NAME is None!"
    assert model_display_name, "Environment variable MODEL_DISPLAY_NAME is None!"
    assert gcs_location, "Environment variable GCS_LOCATION is None!"
    assert model_registry, "Environment variable MODEL_REGISTRY_URI is None!"

    logging.info(f"upload_model: {upload_model}")
    if tf.io.gfile.exists(gcs_location):
        tf.io.gfile.rmtree(gcs_location)
    logging.info(f"Pipeline e2e test artifacts stored in: {gcs_location}")

    if tf.io.gfile.exists(MLMD_SQLLITE):
        tf.io.gfile.remove(MLMD_SQLLITE)

    metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    metadata_connection_config.sqlite.filename_uri = MLMD_SQLLITE
    metadata_connection_config.sqlite.connection_mode = 3
    logging.info("ML metadata store is ready.")

    logging.info(f"gcs_data_location: {gcs_data_location}")
    if tf.io.gfile.exists(gcs_data_location):
        tf.io.gfile.remove(gcs_data_location)
    tf.io.gfile.copy("data/data.csv", gcs_data_location)

    pipeline_root = os.path.join(
        config.ARTIFACT_STORE_URI,
        config.PIPELINE_NAME,
    )
    logging.info(f"pipeline_root: {pipeline_root}")

    runner = LocalDagRunner()

    pipeline = training_pipeline.create_pipeline(
        pipeline_root=pipeline_root,
        metadata_connection_config=metadata_connection_config,
    )

    runner.run(pipeline)

    logging.info(f"Model output: {os.path.join(model_registry, model_display_name)}")
    assert tf.io.gfile.exists(os.path.join(model_registry, model_display_name))
