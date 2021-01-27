import argparse
import csv
import hashlib
import json
from pathlib import Path
import multiprocessing
import glob, os

import pandas as pd
import spacy
from nltk import pos_tag
from joblib import Parallel, delayed
import dask.dataframe as dd

from data import DATA_ROOT as DATA_PATH
from data import WORDS_DATA_PATH, WORDS_FROM_EMBEDDINGS_DATA_PATH
from evaluation import EVALUATION_ROOT as EVALUATION_PATH


nlp = spacy.load("en_core_web_sm")
num_cores = multiprocessing.cpu_count() - 1


def hash_data(word, actual_word_features, possible_word_features):
    return hashlib.sha224(
        f"{word}{actual_word_features}{possible_word_features}".encode("utf-8")
    ).hexdigest()


def validate_train_features(data_source, df):
    def iterate_over_words(word, cnt):
        words_left = len(df["actual_words"].tolist()) - cnt
        print(f"current iteration: {cnt}, words left: {words_left}")
        try:
            actual_word_features = len(words[word])
        except KeyError as e:
            print(e, "keyerror")
            actual_word_features = 0

        try:
            possible_word_features = df.loc[word].sum()
            print(f"possible word features {possible_word_features}")
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

    words_path = {
        "embeddings": WORDS_FROM_EMBEDDINGS_DATA_PATH,
        "common_words": WORDS_DATA_PATH,
    }
    words = {}
    headers = ["Word", "Expected Features", "Actual Features"]
    possible_errors_file = Path(f"{DATA_PATH}/{data_source}_possible_errors.csv")
    processed_words_file = Path(f"{DATA_PATH}/{data_source}_processed_words.txt")

    if not possible_errors_file.is_file():
        with open(possible_errors_file, "a+") as file:
            wr = csv.writer(file, dialect="excel")
            wr.writerow(headers)

    with open(words_path.get(data_source)) as file:
        reader = file.readlines()
        words = json.loads(reader[0])

    processed_words = []
    with open(processed_words_file, "w+") as file:
        processed_words = file.readlines()

    possible_errors = []

    _ = Parallel(n_jobs=num_cores)(
        delayed(iterate_over_words)(word, cnt)
        for cnt, word in enumerate(df["actual_words"].tolist())
    )


def process_train_data(df, data_source, threshold=10):
    print("called process_train_data")
    # remove columns below threshold
    # df_with_threshold = df.loc[:, (df.sum(axis=0) >= threshold)].copy()
    df_with_threshold = df.loc[
        :, (df.iloc[:, 1:].sum(axis=0).compute() >= threshold)
    ].copy()
    print(f"current threshold: {threshold}")
    print(list(df_with_threshold))
    if df_with_threshold.empty:
        print(f"Data frame is empty for threshold: {threshold}")
        exit()
    # create sum of rows
    df_with_threshold["_sum_of_features"] = (
        df_with_threshold.iloc[:, 1:].sum(axis=1).compute()
    )

    # drop words with no features or fishy number of features
    max_features = len(list(df)) - 3
    df_with_threshold.drop(
        df_with_threshold[
            (df_with_threshold._sum_of_features <= 0)
            | (df_with_threshold._sum_of_features >= max_features)
        ].index,
        inplace=True,
    )
    del df_with_threshold["_sum_of_features"]

    feature_metrics = df_with_threshold.sum(axis=0, skipna=True)
    feature_metrics[1:].sort_values(ascending=False).to_csv(
        f"{EVALUATION_PATH}/{data_source}_{threshold}_feature_metrics.csv",
        header=["count_of_words_having_this_feature"],
    )

    df_with_threshold.to_csv(f"{DATA_PATH}/{data_source}_{threshold}_clean_train.csv")

    return f"{DATA_PATH}/{data_source}_{threshold}_clean_train.csv"


def _get_pos(word, is_spacy=True):
    if is_spacy:
        return nlp(word)[0].pos_
    return pos_tag([word])[0][1]


def enrich_data(clean_train, data_source):
    threshold = clean_train.split("_")[-3]
    enriched_data = f"{DATA_PATH}/{data_source}_{threshold}_enriched_data.csv"
    c_df = pd.read_csv(clean_train)
    c_df["POS_TAG"] = c_df["actual_words"].apply(_get_pos)
    c_df.to_csv(enriched_data, index=False)
    return enriched_data


def process_enriched_data(enriched_data, data_source):
    threshold = enriched_data.split("_")[-3]
    final_data = f"{DATA_PATH}/{data_source}_{threshold}_final_data.csv"
    e_df = pd.read_csv(enriched_data)
    dummy = pd.get_dummies(e_df["POS_TAG"], prefix="POS_TAG", drop_first=True)
    del e_df["POS_TAG"]
    e_df = pd.concat([e_df, dummy], axis=1)
    e_df.to_csv(final_data, index=False)
    return final_data


def main(script_args):
    print(script_args.data_source)
    df = None
    print("reading file(s)")
    df = pd.concat(
        map(pd.read_csv, glob.glob(os.path.join("", f"{DATA_PATH}/temp_train/*.csv")))
    )
    print(df.head())
    exit()
    print("setting index")
    # df = df.set_index("actual_words")
    print("done setting index")
    # print(df)
    print("was expensive")

    # validate_train_features(script_args.data_source, df) # seems redundant and perhaps may have an issue.
    # for x in range(5, 100):
    clean_train = process_train_data(df, script_args.data_source, threshold=100)
    enriched_data = enrich_data(clean_train, script_args.data_source)
    process_enriched_data(enriched_data, script_args.data_source)


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
