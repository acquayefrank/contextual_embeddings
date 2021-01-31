import argparse

import multiprocessing


import pandas as pd
import spacy
from nltk import pos_tag

from data import DATA_ROOT as DATA_PATH
from evaluation import EVALUATION_ROOT as EVALUATION_PATH
from .utils import get_logger, generate_uuid


nlp = spacy.load("en_core_web_sm")
num_cores = multiprocessing.cpu_count() - 1
logger = get_logger()


def process_train_data(df, data_source, threshold=100, _logger=None):
    """This function processes, train dataset and removes columns having a threshold below set threshold.

    Args:
        df: The dataframe to be processed
        data_source: The data_source being processed, either embeddings or common_words
        threshold: the threshold for the data
        _logger: Expects a _logger object to be passed

    Returns:
        Path to the saved cleaned data. (Must change the name, sounds wrong)

    """
    if not _logger:
        _logger = logger
    print("About to start processing train data.")

    # remove columns below threshold
    df_with_threshold = df.loc[:, (df.sum(axis=0) >= threshold)].copy()

    # create sum of rows
    df_with_threshold["sum_of_features"] = df_with_threshold.sum(axis=1)

    # drop words with no features
    df_with_threshold.drop(
        df_with_threshold[df_with_threshold.sum_of_features == 0].index, inplace=True
    )
    del df_with_threshold["sum_of_features"]

    feature_metrics = df_with_threshold.sum(axis=0, skipna=True)
    feature_metrics[1:].sort_values(ascending=False).to_csv(
        f"{EVALUATION_PATH}/embeddings_{threshold}_feature_metrics.csv",
        header=["count_of_words_having_this_feature"],
    )

    df_with_threshold.to_csv(f"{DATA_PATH}/{data_source}_{threshold}_clean_train.csv")

    shape = df_with_threshold.shape
    _logger.info(
        f"Total number of words and hyponyms after applying the threshold of {threshold} is {shape}, \
        where the first value is the number of words and the second the number of unique hyponyms\
        i.e number of unique hyponyms that have more than {threshold} words"
    )
    return f"{DATA_PATH}/{data_source}_{threshold}_clean_train.csv"


def _get_pos(word, is_spacy=True):
    """Get the word part of speech tag for a given word

    Args:
        word: The word. whose word embedding is to be found
        is_spacy: Determines if spacy is used or NLTK, default is True hence spacy is used

    Returns:
        A string of the word embedding
    """
    if is_spacy:
        return nlp(word)[0].pos_
    return pos_tag([word])[0][1]


def enrich_data(clean_train, data_source):
    """Add POS_TAG to cleaned data

    Args:
        clean_train: path to cleaned_data
        data_source: The data_source being processed, either embeddings or common_words

    Returns:
        path to enriched file.
    """
    threshold = clean_train.split("_")[-3]
    enriched_data = f"{DATA_PATH}/{data_source}_{threshold}_enriched_data.csv"
    c_df = pd.read_csv(clean_train)
    c_df["POS_TAG"] = c_df["actual_words"].apply(_get_pos)
    c_df.to_csv(enriched_data, index=False)
    return enriched_data


def process_enriched_data(enriched_data, data_source):
    """
    Args:
        enriched_data: A file path to the enriched data
        data_source: The data_source being processed, either embeddings or common_words

    Returns:
        The path to the final dataset

    """
    threshold = enriched_data.split("_")[-3]
    final_data = f"{DATA_PATH}/{data_source}_{threshold}_final_data.csv"
    e_df = pd.read_csv(enriched_data)
    dummy = pd.get_dummies(e_df["POS_TAG"], prefix="POS_TAG", drop_first=True)
    del e_df["POS_TAG"]
    e_df = pd.concat([e_df, dummy], axis=1)
    e_df.to_csv(final_data, index=False)
    return final_data


def main(script_args, _logger=None, run_id=None):
    """
    Args:
        script_args: argparse variables obtained at the time of running script
        _logger: A python logger object for logging run data
        run_id: A uuid4 identifier for tagging runs
    """
    if not _logger:
        _logger = logger

    if not run_id:
        run_id = generate_uuid()

    data_source = script_args.data_source
    print(script_args.data_source)

    if script_args.data_source == "embeddings":
        df = pd.read_csv(f"{DATA_PATH}/word_{script_args.data_source}_train.csv")
    elif script_args.data_source == "common_words":
        df = pd.read_csv(f"{DATA_PATH}/train.csv")
    else:
        df = None

    df.set_index("actual_words", inplace=True)
    clean_train = process_train_data(df, script_args.data_source, 100, _logger)
    enriched_data = enrich_data(clean_train, data_source)
    process_enriched_data(enriched_data, data_source)

    print("Done processing train data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds",
        "--data_source",
        type=str,
        default="embeddings",
        help="The source of data to process, it's either `embeddings` or `common_words`",
    )
    args = parser.parse_args()
    main(args)
