#!/usr/bin/env python
# coding: utf-8
import argparse
import csv
import datetime

import pandas as pd
import torch
from gensim.models import KeyedVectors
from torch.autograd import Variable

from data import DATA_ROOT
from evaluation import EVALUATION_ROOT
from models import MODELS_PATH
from src.models import LogisticRegression, SingleLayeredNN
from src.utils import embeddings, evaluate, get_train_run_parser


def _load_word_embedding_model(
    file=f"../models/fasttext/crawl-300d-2M-subword.vec", word_embedding_type="fasttext"
):
    model = {}
    if file is None:
        file, *ign = embeddings.get("GLOVE_6B_300D")
    print("Loading Model")
    if word_embedding_type == "glove":
        df = pd.read_csv(file, sep=" ", quoting=3, header=None, index_col=0)
        model = {key: val.values for key, val in df.T.items()}
        print(len(model), " words loaded!")
    elif word_embedding_type == "word2vec":
        model = KeyedVectors.load_word2vec_format(file, binary=True)
    elif word_embedding_type == "fasttext":
        model = KeyedVectors.load_word2vec_format(file, binary=False)
    return model


def _get_word_embeddings(word, model):
    try:
        return model[word]
    except KeyError as e:
        pass

    try:
        return model.get(word, None)
    except AttributeError:
        pass

    return None


# determine the supported device
def get_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda:0")
    else:
        _device = torch.device("cpu")  # don't have GPU
    return _device


# convert a df to tensor to be used in pytorch
def df_to_tensor(s_df):
    _device = get_device()
    return Variable(torch.Tensor(s_df.values)).to(_device)


def complex_df_to_tensor(_df):
    _device = get_device()
    temp_x = []
    for index, row in _df.iterrows():
        temp_row = []
        row = row.to_dict()
        g_em = []
        for key in row.keys():
            if key != "Embedding":
                temp_row.append(row[key])
            else:
                g_em = Variable(torch.Tensor(row[key])).to(_device)
        temp_x.append(
            torch.cat([Variable(torch.Tensor(temp_row)).to(_device), g_em], dim=0)
        )

    return torch.stack(temp_x, 0)


def get_data(word, final_data_path, model):
    df = pd.read_csv(final_data_path)
    df["Embedding"] = df["actual_words"].apply(_get_word_embeddings, args=(model,))
    df.dropna(inplace=True)

    x_data = df.loc[:, df.columns == "Embedding"]
    y_data = df.loc[:, df.columns == word]

    y_data = df_to_tensor(y_data)
    x_data = complex_df_to_tensor(x_data)

    return x_data, y_data


def main(script_args):
    run_id = script_args.run_id
    final_data_path = f"{DATA_ROOT}/{run_id}_{script_args.data_source}_{script_args.threshold}_final_data.csv"
    if script_args.all_embeddings:
        word_embeddings = [
            "GLOVE_6B_50D",
            "GLOVE_6B_100D",
            "GLOVE_6B_200D",
            "GLOVE_6B_300D",
            "GLOVE_42B_300D",
            "GLOVE_840B_300D",
            "GLOVE_TWITTER_27B_25D",
            "GLOVE_TWITTER_27B_50D",
            "GLOVE_TWITTER_27B_100D",
            "GLOVE_TWITTER_27B_200D",
            "WORD2VEC_GOOGLE_NEWS_300D",
            "FASTTEXT_CRAWL_SUB",
            "FASTTEXT_CRAWL_VEC_300D",
            "FASTTEXT_WIKI_SUB_300D",
            "FASTTEXT_WIKI_VEC_300D",
        ]
    else:
        word_embeddings = [script_args.word_embeddings]

    df = pd.read_csv(final_data_path)
    words = list(df.columns)
    words.remove("actual_words")
    for embedding_name in word_embeddings:
        em_path, em_dim, em_type = embeddings.get(embedding_name)
        model = _load_word_embedding_model(file=em_path, word_embedding_type=em_type)
        headers = [
            "Training Scenario",
            "Accuracy - Logistic Regression",
            "AUC - Logistic Regression",
            "Accuracy - Single Layered Neural Network",
            "AUC - Single Layered Neural Network",
        ]
        emb_file_name = f'{EVALUATION_ROOT}/{embedding_name}_{script_args.data_source}_test_results_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv'
        with open(emb_file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        for word in words:
            nn_model_path = f"{MODELS_PATH}/{run_id}_trained_models/{word}_{embedding_name}_SingleLayeredNN.pth"
            lr_model_path = f"{MODELS_PATH}/{run_id}_trained_models/{word}_{embedding_name}_LogisticRegression.pth"

            nn_model = SingleLayeredNN(em_dim, em_dim, 1)
            try:
                nn_model.load_state_dict(torch.load(nn_model_path))
            except FileNotFoundError as e:
                print(e)
                continue
            nn_model.eval()

            lr_model = LogisticRegression(em_dim, 1)
            try:
                lr_model.load_state_dict(torch.load(lr_model_path))
            except FileNotFoundError as e:
                print(e)
                continue
            lr_model.eval()

            device = get_device()
            lr_model.to(device)
            nn_model.to(device)

            x_data, y_data = get_data(word, final_data_path, model)

            nn_scores = evaluate(nn_model, x_data, y_data)
            lr_scores = evaluate(lr_model, x_data, y_data)

            results = [
                word,
                lr_scores["_accuracy"],
                lr_scores["_auc"],
                nn_scores["_accuracy"],
                nn_scores["_auc"],
            ]
            with open(emb_file_name, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = get_train_run_parser(parser)
    args = parser.parse_args()
    main(args)
