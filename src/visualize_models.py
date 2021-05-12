import argparse
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import rcParams

from evaluation import EVALUATION_ROOT as EVALUATION_PATH
from models import MODELS_PATH
from src.models import LogisticRegression

from .utils import embeddings, get_trained_models


def main(script_args):
    trained_models_root: str = f"{MODELS_PATH}/{script_args.run_id}_trained_models"
    models = [
        f for f in listdir(trained_models_root) if isfile(join(trained_models_root, f))
    ]
    logistic_models = [
        model for model in models if model.split("_")[-1][:-4] == "LogisticRegression"
    ]
    trained_models = get_trained_models(logistic_models)
    all_named_parameters = {}
    for lg_mdl, t_m in zip(logistic_models, trained_models):
        print(lg_mdl, t_m)
        file_path, dim, embedding_type = embeddings.get(t_m[1])
        print(file_path, dim, embedding_type)
        lg_model = LogisticRegression(dim, 1)
        lg_model.load_state_dict(torch.load(f"{trained_models_root}/{lg_mdl}"))
        lg_model.eval()
        named_parameters = list(lg_model.named_parameters())
        dict_key = f"{t_m[1]}_{dim}"
        weights = named_parameters[0][1].detach().numpy()
        if dict_key not in all_named_parameters:
            all_named_parameters[dict_key] = [(weights, t_m[0])]
        else:
            all_named_parameters[dict_key].append((weights, t_m[0]))

    df = pd.DataFrame(
        {
            "embeddings": [
                key
                for key in all_named_parameters
                for data in all_named_parameters[key]
            ],
            "words": [
                data[1]
                for key in all_named_parameters
                for data in all_named_parameters[key]
            ],
            "weights": [
                data[0]
                for key in all_named_parameters
                for data in all_named_parameters[key]
            ],
        }
    )

    df_m = (
        df.groupby(["embeddings", "words"])["weights"]
        .apply(list)
        .apply(lambda x: np.concatenate(x, axis=0))
    )

    df_m = df_m.unstack(level=0)

    for column in df_m.columns:
        words = [index for index, value in df_m[column].items()]
        weights = [value for index, value in df_m[column].items()]
        weights = np.concatenate(weights)

        # Plot Heatmaps
        _, ax = plt.subplots(figsize=(11, 9))
        sb.heatmap(weights, yticklabels=words)
        ax.set_title(f"{column} HEATMAP")
        plt.savefig(f"{EVALUATION_PATH}/{script_args.run_id}_heatmap_{column}.png")

        # Plot Dendrogram
        plot = sb.clustermap(
            weights, metric="correlation", yticklabels=words, figsize=(15, 13)
        )
        plot.fig.suptitle(f"{column} DENDROGRAM")
        plot.savefig(f"{EVALUATION_PATH}/{script_args.run_id}_dendrogram_{column}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-id",
        "--run_id",
        type=str,
        required=True,
        help="Provide a unique identifier which would be used to track the running of the experiment,\
            in the case where it's not provided one will be generated for you. \
            In order to continue the experiment from when it failed,provide it's unique identifier",
    )
    parser.add_argument(
        "-ds",
        "--data_source",
        type=str,
        default="embeddings",
        help="The source of data to process, it's either `embeddings` or `common_words`",
    )
    args = parser.parse_args()
    main(args)
