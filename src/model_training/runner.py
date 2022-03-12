import os
import logging

from src.model_training import trainer, exporter, defaults


def run_fn(fn_args):
    """Train the model based on given args.
    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    logging.info("Runner started...")
    logging.info(f"fn_args: {fn_args}")
    logging.info("")

    logging.info("Runner executing trainer...")
    trained_model = trainer.train(fn_args)

    logging.info("Runner executing exporter...")
    exporter.export_serving_model(
        trained_model,
        fn_args.serving_model_dir,
        fn_args.schema_path,
    )
    logging.info("Runner completed.")
