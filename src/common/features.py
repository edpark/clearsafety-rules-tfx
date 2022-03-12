
FEATURE_NAMES = [
    'feature_1', 
    'feature_2', 
    'feature_3',
    'feature_4', 
    'feature_5'
]

TARGET_FEATURE_NAME = "label"

TARGET_LABELS = ["label"]

NUMERICAL_FEATURE_NAMES = [
    
]

EMBEDDING_CATEGORICAL_FEATURES = {
    
}

ONEHOT_CATEGORICAL_FEATURE_NAMES = []


def transformed_name(key: str) -> str:
    """Generate the name of the transformed feature from original name."""
    return f"{key}_xf"


def original_name(key: str) -> str:
    """Generate the name of the original feature from transformed name."""
    return key.replace("_xf", "")


def vocabulary_name(key: str) -> str:
    """Generate the name of the vocabulary feature from original name."""
    return f"{key}_vocab"


def categorical_feature_names() -> list:
    return (
        list(EMBEDDING_CATEGORICAL_FEATURES.keys()) + ONEHOT_CATEGORICAL_FEATURE_NAMES
    )


def generate_explanation_config():
    explanation_config = {
        "inputs": {},
        "outputs": {},
        "params": {"sampled_shapley_attribution": {"path_count": 10}},
    }

    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERICAL_FEATURE_NAMES:
            explanation_config["inputs"][feature_name] = {
                "input_tensor_name": feature_name,
                "modality": "numeric",
            }
        else:
            explanation_config["inputs"][feature_name] = {
                "input_tensor_name": feature_name,
                "encoding": 'IDENTITY',
                "modality": "categorical",
            }

    explanation_config["outputs"] = {"scores": {"output_tensor_name": "scores"}}

    return explanation_config
