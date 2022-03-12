import tensorflow as tf

from src.common import features


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def get_dataset(file_pattern, feature_spec, batch_size=200):
    """Generates features and label for tuning/training.
    Args:
      file_pattern: input tfrecord file pattern.
      feature_spec: a dictionary of feature specifications.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch
    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        label_key=features.TARGET_FEATURE_NAME,
        reader=_gzip_reader_fn,
        num_epochs=1,
        drop_final_batch=True,
    )

    return dataset

def get_csv_dataset(file_pattern):
    csv_ds = tf.data.experimental.make_csv_dataset(
        file_pattern,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        label_name=features.TARGET_FEATURE_NAME,)
    
    return csv_ds