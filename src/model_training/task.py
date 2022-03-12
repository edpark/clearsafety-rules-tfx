import os
import sys
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse

from google.cloud import aiplatform as vertex_ai

from src.model_training import defaults, trainer, exporter


dirname = os.path.dirname(__file__)
dirname = dirname.replace("/model_training", "")
RAW_SCHEMA_LOCATION = os.path.join(dirname, "raw_schema/schema.pbtxt")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        default=os.getenv("AIP_MODEL_DIR"),
        type=str,
    )

    parser.add_argument(
        "--log-dir",
        default=os.getenv("AIP_TENSORBOARD_LOG_DIR"),
        type=str,
    )

    parser.add_argument(
        "--train-data-dir",
        type=str,
    )

    parser.add_argument(
        "--eval-data-dir",
        type=str,
    )

    parser.add_argument("--project", type=str)
    parser.add_argument("--region", type=str)
    parser.add_argument("--staging-bucket", type=str)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--run-name", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    if args.experiment_name:
        vertex_ai.init(
            project=args.project,
            staging_bucket=args.staging_bucket,
            experiment=args.experiment_name,
        )

        logging.info(f"Using Vertex AI experiment: {args.experiment_name}")

        run_id = args.run_name
        if not run_id:
            run_id = f"run-gcp-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        vertex_ai.start_run(run_id)
        logging.info(f"Run {run_id} started.")

    trained_model = trainer.train(
        fn_args=None,
        csv_data_dir=args.train_data_dir,
    )

    val_loss, val_accuracy = trainer.evaluate(
        trained_model,
        args.eval_data_dir,
    )
    
    
    # Log metrics in Vertex Experiments.
    logging.info(f'Logging metrics to Vertex Experiments...')
    if args.experiment_name:
        vertex_ai.log_metrics({"val_loss": val_loss, "val_accuracy": val_accuracy})

    try:
        exporter.export_serving_model(
            trained_model=trained_model,
            serving_model_dir=args.model_dir,
            raw_schema_location=RAW_SCHEMA_LOCATION,
            tft_output_dir=args.tft_output_dir,
        )
    except:
        # Swallow Ignored Errors while exporting the model.
        pass


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Python Version = {sys.version}")
    logging.info(f"TensorFlow Version = {tf.__version__}")
    logging.info(f'TF_CONFIG = {os.environ.get("TF_CONFIG", "Not found")}')
    logging.info(f"DEVICES = {device_lib.list_local_devices()}")
    logging.info(f"Task started...")
    main()
    logging.info(f"Task completed.")
