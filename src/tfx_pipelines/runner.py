import os
from kfp.v2.google.client import AIPlatformClient
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner


from src.tfx_pipelines import config, training_pipeline, prediction_pipeline
from src.model_training import defaults


def compile_training_pipeline(pipeline_definition_file):

    pipeline_root = os.path.join(
        config.ARTIFACT_STORE_URI,
        config.PIPELINE_NAME,
    )

    managed_pipeline = training_pipeline.create_pipeline(
        pipeline_root=pipeline_root,
    )

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            default_image=config.TFX_IMAGE_URI
        ),
        output_filename=pipeline_definition_file,
    )

    return runner.run(managed_pipeline, write_out=True)


def compile_prediction_pipeline(pipeline_definition_file):

    pipeline_root = os.path.join(
        config.ARTIFACT_STORE_URI,
        config.PIPELINE_NAME,
    )

    managed_pipeline = prediction_pipeline.create_pipeline(
        pipeline_root=pipeline_root,
    )

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            default_image=config.TFX_IMAGE_URI
        ),
        output_filename=pipeline_definition_file,
    )

    return runner.run(managed_pipeline, write_out=True)


def submit_pipeline(pipeline_definition_file):

    pipeline_client = AIPlatformClient(project_id=config.PROJECT, region=config.REGION)
    pipeline_client.create_run_from_job_spec(pipeline_definition_file)
