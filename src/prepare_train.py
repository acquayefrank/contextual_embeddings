import argparse
import asyncio
import csv
import datetime
import json
import multiprocessing
import re
import sys
import time
from pathlib import Path
from typing import Set

import aiohttp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.core._multiarray_umath import ndarray

from data import DATA_ROOT

from .utils import _load_word_embedding_model, embeddings

num_cores = multiprocessing.cpu_count() - 1


def get_words_from_embeddings():
    min_number_of_words_in_all_embeddings = 0
    unprocessed_words = []

    for index, embedding in enumerate(embeddings):
        print(embedding, " :being processed ", index, " :embedding count")
        file_path, num_features, word_embedding_type = embeddings.get(embedding)
        model = _load_word_embedding_model(
            file=file_path, word_embedding_type=word_embedding_type
        )

        if word_embedding_type in ["word2vec", "fasttext"]:
            keys = set(str(key).lower() for key in model.vocab.keys())
        elif word_embedding_type == "glove":
            keys = set(str(key).lower() for key in model.keys())
        else:
            keys = set()

        if len(keys) > min_number_of_words_in_all_embeddings:
            min_number_of_words_in_all_embeddings = len(keys)
        unprocessed_words.append(keys)

    _words = set.intersection(*unprocessed_words)

    print(
        f"The minimum number of words in word embeddings is: {min_number_of_words_in_all_embeddings}"
    )
    return list(_words)


def get_words_with_hyponyms(_words):
    print("called get words with hyponyms")
    words_with_hyponyms = {}

    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    async def get_hyponym(session, word):
        print("called get hyponyms")
        async with session.get(
            f"http://api.datamuse.com/words?rel_spc={word}"
        ) as response:
            try:
                words_with_hyponyms[word] = await response.json()
                print("Read {0} from {1}".format(response.content_length, word))
            except Exception as e:
                print(e)
                pass

    async def get_all_hyponyms(words):
        print("called get all hyponyms")
        async with aiohttp.ClientSession() as session:
            tasks = []
            for word in words:
                task = asyncio.ensure_future(get_hyponym(session, word))
                tasks.append(task)
            await asyncio.gather(*tasks, return_exceptions=True)

    words_chunks = list(chunks(_words, 100000))

    for word_chunks in words_chunks:
        start_time = time.time()
        asyncio.get_event_loop().run_until_complete(get_all_hyponyms(word_chunks))
        duration = time.time() - start_time
        print(f"Words processed {len(_words)} in {duration} seconds")

    return words_with_hyponyms


def save_words_with_hyponyms(_words, file_name):
    with open(f"{DATA_ROOT}/{file_name}", "w") as file:
        file.write(json.dumps(_words))  # use `json.loads` to do the reverse
    return True


def load_words(filename):
    if filename == "":
        return {}
    with open(filename, "r") as reader:
        words = reader.readlines()
        words = json.loads(words[0])
    return words


def get_max_length_from_words(_words):
    max_length = 0
    for index, word in enumerate(_words):
        len_words = len(_words[word])
        if len_words > max_length:
            max_length = len_words
    return max_length


def get_words_with_extracted_features(words, max_length):
    def gen_words_with_extracted_features(key, word):
        print(f"{key}: has been queued for processing")
        features = [data["word"] for data in word]
        length_left = max_length - len(features)
        features = features + [" " for _ in range(length_left)]
        return key, sorted(features)

    results = Parallel(n_jobs=num_cores)(
        delayed(gen_words_with_extracted_features)(key, word)
        for key, word in words.items()
    )
    words_with_extracted_features = {key: feature for key, feature in results}

    return words_with_extracted_features


def save_train_features(words_with_extracted_features_path, words, temp_file_path):
    print("make sure temp_train folder is empty before proceding")
    wwef_df = pd.read_csv(words_with_extracted_features_path, index_col=0)

    def gen_features(word):
        features = [data["word"] for data in word]
        return features

    all_features = Parallel(n_jobs=num_cores)(
        delayed(gen_features)(word) for _, word in words.items()
    )
    final_features = [value for feature in all_features for value in feature]

    all_features = set(final_features)

    def process_final_data(args):
        # write to work with chucks
        word, c_row = args
        print("processing final data")
        _fwe_df = pd.DataFrame(
            0,
            index=np.arange(1, dtype=np.byte),
            columns=sorted(list(all_features)),
            dtype=np.byte,
        )
        _fwe_df.insert(loc=0, column="actual_words", value=word)
        _fwe_df.set_index("actual_words", inplace=True)

        for _index, _row in _fwe_df.iterrows():
            for _data in c_row:
                if _row.get(_data) is not None:
                    _fwe_df.at[_index, _data] = 1
        # It is assumed that at any point in time, temp will contain only data for current run.
        _file_name = f"{DATA_ROOT}/temp_train/{script_args.data_source}train_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
        _fwe_df.to_csv(_file_name)
        return _file_name

    file_names = []
    chunks = np.array_split(wwef_df, 1000000)
    for cnt, chunk in enumerate(chunks):
        file_name = Parallel(n_jobs=num_cores)(
            delayed(process_final_data)((index, row)) for index, row in chunk.iterrows()
        )
        file_names += file_name

    with open(temp_file_path, "w",) as f:
        for item in file_names:
            f.write("%s\n" % item)


def main(script_args):
    file_name = ""
    if script_args.data_source == "embeddings":

        # https://stackoverflow.com/a/15063941/1817530
        max_int = sys.maxsize
        while True:
            # decrease the max_int value by factor 10
            # as long as the OverflowError occurs.

            try:
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)

        words_from_embeddings = f"{DATA_ROOT}/words_from_embeddings.csv"
        if not Path(words_from_embeddings).is_file():
            words = get_words_from_embeddings()
            with open(words_from_embeddings, "w", encoding="utf-8") as result_file:
                wr = csv.writer(result_file, dialect="excel")
                wr.writerows([words])
        else:
            with open(words_from_embeddings, "r", encoding="utf-8") as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONE)
                words = list(reader)
        words = words[0]
        assert (
            len(words)
            <= 2_702_150  # number is hard coded since The minimum number of words in all word embeddings is 2_702_150
        ), "The minimum number of words in all word embeddings is 2_702_150 hence intersection should be less"
        words = re.findall(r"'(\w+)'", str(words))
        file_name = f"{DATA_ROOT}/_words_from_word_embeddings_with_hyponyms.txt"
        if not Path(file_name).is_file():
            words_with_hyponyms = get_words_with_hyponyms(words)
            save_words_with_hyponyms(
                words_with_hyponyms, "_words_from_word_embeddings_with_hyponyms.txt"
            )

        print(len(words))
        exit()

    elif script_args.data_source == "common_words":

        with open(f"{DATA_ROOT}/corncob_lowercase.txt", "r") as reader:
            content = reader.readlines()
            content = [line.strip() for line in content]
        file_name = f"{DATA_ROOT}/_words_with_hyponyms.txt"
        if not Path(file_name).is_file():
            words_with_hyponyms = get_words_with_hyponyms(content)
            save_words_with_hyponyms(words_with_hyponyms, "_words_with_hyponyms.txt")

    words_with_extracted_features_path = (
        f"{DATA_ROOT}/words_with_extracted_features_{script_args.data_source}.csv"
    )
    words = load_words(file_name)
    if not Path(words_with_extracted_features_path).is_file():
        max_length = get_max_length_from_words(words)
        words_with_extracted_features = get_words_with_extracted_features(
            words, max_length
        )
        wwef_df = pd.DataFrame.from_dict(words_with_extracted_features, orient="index")
        wwef_df.to_csv(words_with_extracted_features_path)

    temp_file_path = f"{DATA_ROOT}/temp_train_{script_args.data_source}_{datetime.datetime.now().strftime('%Y_%m')}.txt"
    if not Path(temp_file_path).is_file():
        save_train_features(words_with_extracted_features_path, words, temp_file_path)


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
