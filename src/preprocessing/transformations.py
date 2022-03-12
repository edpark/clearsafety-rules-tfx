import tensorflow as tf
import tensorflow_transform as tft

from src.common import features


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      the inputs (no transformation needed for Decision Forests and particularly this dataset)
    """

    return inputs
