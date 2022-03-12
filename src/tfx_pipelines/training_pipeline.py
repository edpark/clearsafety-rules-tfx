import os
import sys
import logging
import json

import tensorflow_model_analysis as tfma

from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2, transform_pb2, pusher_pb2
from tfx.types import Channel, standard_artifacts
from tfx.orchestration import pipeline, data_types
from tfx.dsl.components.common.importer import Importer
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.v1.extensions.google_cloud_big_query import BigQueryExampleGen
from tfx.v1.extensions.google_cloud_ai_platform import Trainer as VertexTrainer 
from tfx.v1.components import (
    CsvExampleGen,
    StatisticsGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from src.tfx_pipelines import config
from src.tfx_pipelines import components as custom_components
from src.common import features, datasource_utils

RAW_SCHEMA_DIR = "src/raw_schema"
TRANSFORM_MODULE_FILE = "src/preprocessing/transformations.py"
TRAIN_MODULE_FILE = "src/model_training/runner.py"


def create_pipeline(
    pipeline_root: str,
    metadata_connection_config: metadata_store_pb2.ConnectionConfig = None,
):
    logging.info(f"create_pipeline - pipeline_root: {pipeline_root}, metadata_connection_config: {metadata_connection_config}")

    # Hyperparameter generation.
    # hyperparams_gen = custom_components.hyperparameters_gen(
    #     num_epochs=num_epochs,
    #     batch_size=batch_size,
    #     learning_rate=learning_rate,
    #     hidden_units=hidden_units,
    # ).with_id("HyperparamsGen")

    # Train example generation.
    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='*'),
        example_gen_pb2.Input.Split(name='eval', pattern='*')
    ])

    train_example_gen = CsvExampleGen(
        input_base=datasource_utils.get_training_csv_source(config.GCS_LOCATION),
        input_config=input_config
    ).with_id("TrainDataGen")

    # Schema importer.
    schema_importer = Importer(
        source_uri=RAW_SCHEMA_DIR,
        artifact_type=standard_artifacts.Schema,
    ).with_id("SchemaImporter")

    # Statistics generation.
    statistics_gen = StatisticsGen(examples=train_example_gen.outputs["examples"]).with_id(
        "StatisticsGen"
    )

    # Example validation.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_importer.outputs["result"],
    ).with_id("ExampleValidator")

    # Data transformation.
    transform = Transform(
        examples=train_example_gen.outputs["examples"],
        schema=schema_importer.outputs["result"],
        module_file=TRANSFORM_MODULE_FILE,
        # This is a temporary workaround to run on Dataflow.
        force_tf_compat_v1=config.BEAM_RUNNER == "DataflowRunner",
        splits_config=transform_pb2.SplitsConfig(
            analyze=["train"], transform=["train", "eval"]
        ),
    ).with_id("DataTransformer")

    # Add dependency from example_validator to transform.
    transform.add_upstream_node(example_validator)

    # Get the latest model to warmstart
    warmstart_model_resolver = Resolver(
        strategy_class=latest_artifacts_resolver.LatestArtifactsResolver,
        latest_model=Channel(type=standard_artifacts.Model),
    ).with_id("WarmstartModelResolver")

    # Model training.
    trainer = Trainer(
        module_file=TRAIN_MODULE_FILE,
        examples=transform.outputs["transformed_examples"],
        schema=schema_importer.outputs["result"],
        base_model=warmstart_model_resolver.outputs["latest_model"],
        transform_graph=transform.outputs["transform_graph"],
    ).with_id("ModelTrainer")
    
    if config.TRAINING_RUNNER == "vertex":
        trainer = VertexTrainer(
            module_file=TRAIN_MODULE_FILE,
            examples=transform.outputs["transformed_examples"],
            schema=schema_importer.outputs["result"],
            base_model=warmstart_model_resolver.outputs["latest_model"],
            transform_graph=transform.outputs["transform_graph"],
            custom_config=config.VERTEX_TRAINING_CONFIG
        ).with_id("ModelTrainer")
        

    # Get the latest blessed model (baseline) for model validation.
    baseline_model_resolver = Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=standard_artifacts.Model),
        model_blessing=Channel(type=standard_artifacts.ModelBlessing),
    ).with_id("BaselineModelResolver")

    # Prepare evaluation config.
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="serving_tf_example",
                label_key=features.TARGET_FEATURE_NAME,
                prediction_key="probabilities",
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": float(config.ACCURACY_THRESHOLD)}
                            ),
                            # Change threshold will be ignored if there is no
                            # baseline model resolved from MLMD (first run).
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )

    # Model evaluation.
    logging.info(f"trainer.outputs: {trainer.outputs}, baseline_model_resolver.outputs: {baseline_model_resolver.outputs}")
    evaluator = Evaluator(
        examples=train_example_gen.outputs["examples"],
        example_splits=["train"],
        model=trainer.outputs["model"],
        baseline_model=baseline_model_resolver.outputs["model"],
        eval_config=eval_config,
        schema=schema_importer.outputs["result"],
    ).with_id("ModelEvaluator")
    if int(config.USE_EVALUATOR):
        evaluator = evaluator.with_beam_pipeline_args(config.BEAM_DATAFLOW_PIPELINE_ARGS + ["--sdk_location=container", "--experiments=use_runner_v2", f"--sdk_container_image={config.CUSTOM_DATAFLOW_IMAGE_URI}"])

    exported_model_location = os.path.join(
        config.MODEL_REGISTRY_URI, config.MODEL_DISPLAY_NAME
    )
    push_destination = pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=exported_model_location
        )
    )

    # Push custom model to model registry.
    pusher = Pusher(
        model=trainer.outputs["model"],
        # model_blessing=evaluator.outputs["blessing"],
        push_destination=push_destination,
    ).with_id("ModelPusher")
    
    # Upload custom trained model to Vertex AI.
    labels = {
        "dataset_name": config.DATASET_DISPLAY_NAME,
        "pipeline_name": config.PIPELINE_NAME,
        "pipeline_root": pipeline_root
    }
    labels = json.dumps(labels)
    explanation_config = json.dumps(features.generate_explanation_config())
    
    vertex_model_uploader = custom_components.vertex_model_uploader(
        project=config.PROJECT,
        region=config.REGION,
        model_display_name=config.MODEL_DISPLAY_NAME,
        pushed_model_location=exported_model_location,
        serving_image_uri=config.SERVING_IMAGE_URI,
        # model_blessing=evaluator.outputs["blessing"],
        explanation_config=explanation_config,
        labels=labels
    ).with_id("VertexUploader")

    pipeline_components = [
        train_example_gen,
        statistics_gen,
        schema_importer,
        example_validator,
        transform,
        warmstart_model_resolver,
        trainer,
        baseline_model_resolver,
        pusher,
    ]

    if int(config.USE_EVALUATOR):
        pipeline_components.append(evaluator)

    if int(config.UPLOAD_MODEL):
        pipeline_components.append(vertex_model_uploader)
        # Add dependency from pusher to aip_model_uploader.
        vertex_model_uploader.add_upstream_node(pusher)

    logging.info(
        f"Pipeline components: {[component.id for component in pipeline_components]}"
    )

    beam_pipeline_args = config.BEAM_DIRECT_PIPELINE_ARGS
    if config.BEAM_RUNNER == "DataflowRunner":
        beam_pipeline_args = config.BEAM_DATAFLOW_PIPELINE_ARGS

    logging.info(f"Beam pipeline args: {beam_pipeline_args}")

    return pipeline.Pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=pipeline_components,
        beam_pipeline_args=beam_pipeline_args,
        metadata_connection_config=metadata_connection_config,
        enable_cache=int(config.ENABLE_CACHE)
    )
