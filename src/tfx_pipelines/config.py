import os
from tfx import v1 as tfx

PROJECT = os.getenv("PROJECT", "")
REGION = os.getenv("REGION", "")
GCS_LOCATION = os.getenv("GCS_LOCATION", "")

ARTIFACT_STORE_URI = os.path.join(GCS_LOCATION, "tfx_artifacts")
MODEL_REGISTRY_URI = os.getenv(
    "MODEL_REGISTRY_URI",
    os.path.join(GCS_LOCATION, "model_registry"),
)

DATASET_DISPLAY_NAME = os.getenv("DATASET_DISPLAY_NAME", "tfdf-rules")
MODEL_DISPLAY_NAME = os.getenv(
    "MODEL_DISPLAY_NAME", f"{DATASET_DISPLAY_NAME}-classifier"
)
PIPELINE_NAME = os.getenv("PIPELINE_NAME", f"{MODEL_DISPLAY_NAME}-train-pipeline")

ML_USE_COLUMN = "ml_use"
EXCLUDE_COLUMNS = ",".join([""])
TRAIN_LIMIT = os.getenv("TRAIN_LIMIT", "0")
TEST_LIMIT = os.getenv("TEST_LIMIT", "0")
SERVE_LIMIT = os.getenv("SERVE_LIMIT", "0")

NUM_TRAIN_SPLITS = os.getenv("NUM_TRAIN_SPLITS", "1")
NUM_EVAL_SPLITS = os.getenv("NUM_EVAL_SPLITS", "1")
ACCURACY_THRESHOLD = os.getenv("ACCURACY_THRESHOLD", "1.0")

USE_KFP_SA = os.getenv("USE_KFP_SA", "False")

TFX_IMAGE_URI = os.getenv(
    "TFX_IMAGE_URI", f"gcr.io/{PROJECT}/{DATASET_DISPLAY_NAME}-tfx:latest"
)
CUSTOM_DATAFLOW_IMAGE_URI = os.getenv(
    "CUSTOM_DATAFLOW_IMAGE_URI", f"gcr.io/{PROJECT}/ml-dataflow-image:latest"
)

BEAM_RUNNER = os.getenv("BEAM_RUNNER", "DirectRunner")
BEAM_DIRECT_PIPELINE_ARGS = [
    f"--project={PROJECT}",
    f"--temp_location={os.path.join(GCS_LOCATION, 'temp')}",
]
BEAM_DATAFLOW_PIPELINE_ARGS = [
    f"--project={PROJECT}",
    f"--temp_location={os.path.join(GCS_LOCATION, 'temp')}",
    f"--region={REGION}",
    f"--runner={BEAM_RUNNER}",
]

# "--sdk_location=container",
# "--experiments=use_runner_v2",
# f"--sdk_container_image={TFX_IMAGE_URI}",

TRAINING_RUNNER = os.getenv("TRAINING_RUNNER", "local")
VERTEX_TRAINING_ARGS = {
    'project': PROJECT,
    'worker_pool_specs': [{
        'machine_spec': {
            'machine_type': 'n1-standard-4',
#             'accelerator_type': 'NVIDIA_TESLA_K80',
#             'accelerator_count': 1
        },
        'replica_count': 1,
        'container_spec': {
            'image_uri': TFX_IMAGE_URI,
        },
    }],
}
VERTEX_TRAINING_CONFIG = {
    tfx.extensions.google_cloud_ai_platform.ENABLE_UCAIP_KEY: True,
    tfx.extensions.google_cloud_ai_platform.UCAIP_REGION_KEY: REGION,
    tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: VERTEX_TRAINING_ARGS,
    'use_gpu': False,
}

# Change this to the location where you've uploaded the ml6team's Tensorflow Serving image
SERVING_IMAGE_URI = ''

BATCH_PREDICTION_BEAM_ARGS = {
    "runner": f"{BEAM_RUNNER}",
    "temporary_dir": os.path.join(GCS_LOCATION, "temp"),
    "gcs_location": os.path.join(GCS_LOCATION, "temp"),
    "project": PROJECT,
    "region": REGION,
    "setup_file": "./setup.py",
}
BATCH_PREDICTION_JOB_RESOURCES = {
    "machine_type": "n1-standard-2",
    #'accelerator_count': 1,
    #'accelerator_type': 'NVIDIA_TESLA_T4'
    "starting_replica_count": 1,
    "max_replica_count": 10,
}
DATASTORE_PREDICTION_KIND = f"{MODEL_DISPLAY_NAME}-predictions"

ENABLE_CACHE = os.getenv("ENABLE_CACHE", "0")
UPLOAD_MODEL = os.getenv("UPLOAD_MODEL", "1")
USE_EVALUATOR = os.getenv("USE_EVALUATOR", "0")

os.environ["PROJECT"] = PROJECT
os.environ["PIPELINE_NAME"] = PIPELINE_NAME
os.environ["TFX_IMAGE_URI"] = TFX_IMAGE_URI
os.environ["CUSTOM_DATAFLOW_IMAGE_URI"] = CUSTOM_DATAFLOW_IMAGE_URI
os.environ["MODEL_REGISTRY_URI"] = MODEL_REGISTRY_URI
