import json
import os
import parsect.constants as constants
from parsect.clients.parsect_inference import ParsectInference
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.modules.bow_encoder import BOW_Encoder
import torch.nn as nn

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]


def get_glove_emb_linear_classifier_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    NUM_CLASSES = config["NUM_CLASSES"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    # it does not matter what random embeddings you have here
    # it will be filled with the learnt parameters while loading the model
    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION)

    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedding=embedding,
        dropout_value=0.0,
        aggregation_type="sum",
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIMENSION,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    parsect_inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(
        OUTPUT_DIR, "bow_glove_emb_lc_3kw_10ml_50d_10e_1e-3lr"
    )
    inference_client = get_glove_emb_linear_classifier_infer(experiment_dirname)
