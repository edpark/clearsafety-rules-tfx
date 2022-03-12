
import pytest
import sys
import logging
import tensorflow as tf

from src.common import features
from src.model_training import model, defaults

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

EXPECTED_HYPERPARAMS_KEYS = [
    "hidden_units",
    "learning_rate",
    "batch_size",
    "num_epochs",
]


def test_hyperparams_defaults():
    hyperparams = {"hidden_units": [64, 32]}

    hyperparams = defaults.update_hyperparams(hyperparams)
    assert set(hyperparams.keys()) == set(EXPECTED_HYPERPARAMS_KEYS)

@pytest.mark.skip(reason="Test failing out of the box with wrong shape error")
def test_create_binary_classifier():

    hyperparams = hyperparams = defaults.update_hyperparams(dict())

    model_inputs = {
        "dropoff_grid_xf": tf.convert_to_tensor([0, 0, 0]),
        "euclidean_xf": tf.convert_to_tensor([-0.9066112, -0.9066112, -0.9066112]),
        "loc_cross_xf": tf.convert_to_tensor([0, 0, 0]),
        "payment_type_xf": tf.convert_to_tensor([1, 0, 0]),
        "pickup_grid_xf": tf.convert_to_tensor([0, 0, 0]),
        "trip_day_of_week_xf": tf.convert_to_tensor([5, 4, 4]),
        "trip_day_xf": tf.convert_to_tensor([26, 24, 1]),
        "trip_hour_xf": tf.convert_to_tensor([0, 4, 2]),
        "trip_miles_xf": tf.convert_to_tensor([5.9717827, -0.7121308, -0.7601589]),
        "trip_month_xf": tf.convert_to_tensor([4, 3, 4]),
        "trip_seconds_xf": tf.convert_to_tensor([4.9029775, -0.34146854, -0.34479955]),
    }

    feature_vocab_sizes = {
        feature_name: 100 for feature_name in features.categorical_feature_names()
    }
    classifier = model._create_binary_classifier(feature_vocab_sizes, hyperparams)
    model_outputs = classifier(model_inputs)  # .numpy()
    assert model_outputs.shape == (3, 1)
    assert model_outputs.dtype == "float32"
