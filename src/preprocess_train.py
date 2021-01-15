import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from nltk import pos_tag

from data import DATA_ROOT as DATA_PATH
from data import TRAIN_DATA_PATH, WORDS_DATA_PATH
from evaluation import EVALUATION_ROOT as EVALUATION_PATH
from models import GLOVE_6B_50D

df = pd.read_csv(TRAIN_DATA_PATH, index_col="actual_words")
nlp = spacy.load("en_core_web_sm")


def _load_glove_model(File=GLOVE_6B_50D):
    print("Loading Glove Model")
    gloveModel = {}
    with open(File, "r", encoding="utf8") as f:
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array(
                [float(value) for value in splitLines[1:]]
            ).tolist()
            gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel


GLOVE_MODEL = _load_glove_model()


def hash_data(word, actual_word_features, possible_word_features):
    return hashlib.sha224(
        f"{word}{actual_word_features}{possible_word_features}".encode("utf-8")
    ).hexdigest()


def validate_train_features():

    words = {}
    headers = ["Word", "Expected Features", "Actual Features"]
    possible_errors_file = Path(DATA_PATH / "possible_errors.csv")
    processed_words_file = Path(DATA_PATH / "processed_words.txt")

    if not possible_errors_file.is_file():
        with open(possible_errors_file, "a+") as file:
            wr = csv.writer(file, dialect="excel")
            wr.writerow(headers)

    with open(WORDS_DATA_PATH) as file:
        reader = file.readlines()
        words = json.loads(reader[0])

    processed_words = []
    with open(processed_words_file, "a+") as file:
        processed_words = file.readlines()

    possible_errors = []
    for word in words:
        actual_word_features = len(words[word])
        try:
            possible_word_features = df.loc[word].sum()
        except KeyError as e:
            print(e)
            possible_word_features = 0
        hash_string = hash_data(word, actual_word_features, possible_word_features)

        if hash_string not in processed_words:

            if actual_word_features != possible_word_features:
                possible_errors.append(
                    [word, actual_word_features, possible_word_features]
                )
                with open(possible_errors_file, "a+") as file:
                    wr = csv.writer(file, dialect="excel")
                    wr.writerows(possible_errors)

            with open(processed_words_file, "a") as file:
                file.write(f"{hash_string}\n")

            print(word, actual_word_features, possible_word_features)


def process_train_data(threshold=100):

    # remove columns below threshold
    df_with_threshold = df.loc[:, (df.sum(axis=0) >= threshold)].copy()

    # create sum of rows
    df_with_threshold["_sum_of_features"] = df_with_threshold.sum(axis=1)

    # drop words with no features
    df_with_threshold.drop(
        df_with_threshold[df_with_threshold._sum_of_features <= 0].index, inplace=True
    )
    del df_with_threshold["_sum_of_features"]

    feature_metrics = df_with_threshold.sum(axis=0, skipna=True)
    feature_metrics[1:].sort_values(ascending=False).to_csv(
        f"{EVALUATION_PATH}/feature_metrics.csv",
        header=["count_of_words_having_this_feature"],
    )

    df_with_threshold.to_csv(f"{DATA_PATH}/clean_train.csv")

    return f"{DATA_PATH}/clean_train.csv"


def _get_pos(word, is_spacy=True):
    if is_spacy:
        return nlp(word)[0].pos_
    return pos_tag([word])[0][1]


def _get_word_embeddings(word):
    return GLOVE_MODEL.get(word, None)


def enrich_data(clean_train):
    enriched_data = f"{DATA_PATH}/enriched_data.csv"
    c_df = pd.read_csv(clean_train)
    c_df["POS_TAG"] = c_df["actual_words"].apply(_get_pos)
    c_df["GLOVE.6B"] = c_df["actual_words"].apply(_get_word_embeddings)
    c_df.to_csv(enriched_data, index=False)
    return enriched_data


def process_enriched_data(enriched_data):
    final_data = f"{DATA_PATH}/final_data.csv"
    e_df = pd.read_csv(enriched_data)
    e_df["GLOVE.6B"].replace("", np.nan, inplace=True)
    e_df.dropna(subset=["GLOVE.6B"], inplace=True)
    dummy = pd.get_dummies(e_df["POS_TAG"], prefix="POS_TAG", drop_first=True)
    del e_df["POS_TAG"]
    e_df = pd.concat([e_df, dummy], axis=1)
    e_df.to_csv(final_data, index=False)
    return final_data


def main(script_args):
    validate_train_features()
    clean_train = process_train_data()
    enriched_data = enrich_data(clean_train)
    process_enriched_data(enriched_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds",
        "--data_source",
        type=str,
        default="embeddings",
        help="The source of data to process, it's either `embeddings` or `common_words`",
    )
    script_args = parser.parse_args()
    main(script_args)
