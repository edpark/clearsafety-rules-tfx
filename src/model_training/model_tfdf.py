import tensorflow as tf
import tensorflow_decision_forests as tfdf


def create_decision_forest() -> tf.keras.Model:
    model = tfdf.keras.RandomForestModel(
        categorical_algorithm='CART',
        min_examples=1
    )
    model.compile(metrics=["accuracy"])

    return model