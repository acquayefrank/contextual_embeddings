import argparse
import asyncio
import csv
import json
import multiprocessing
import re
import sys
import time
from argparse import Namespace
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd

from data import DATA_ROOT as DATA_PATH

from .utils import _load_word_embedding_model, embeddings, generate_uuid, get_logger

UUID = generate_uuid()
num_cores = multiprocessing.cpu_count() - 1


def get_words_from_embeddings(_logger=None):
    """Gets an intersection of words from all word embeddings available.

    This method reads in all word embeddings on the system and finds an intersection of common words.
    
    Args:
        _logger: Expects a _logger object to be passed

    Returns:
        An intersected list of words from all word embeddings
    """
    if not _logger:
        _logger = get_logger(run_id=UUID)

    min_number_of_words_in_all_embeddings = 0
    word_emb = ""
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

        if index == 0:
            min_number_of_words_in_all_embeddings = len(keys)
            word_emb = embedding
            print(
                f"Current minimum number of words in word embeddings is: {min_number_of_words_in_all_embeddings}",
                f"Current word embedding that has the minimum number of words is {word_emb}",
            )

        if len(keys) < min_number_of_words_in_all_embeddings:
            min_number_of_words_in_all_embeddings = len(keys)
            word_emb = embedding
            print(
                f"Current minimum number of words in word embeddings is: {min_number_of_words_in_all_embeddings}",
                f"Current word embedding that has the minimum number of words is {word_emb}",
            )
        unprocessed_words.append(keys)

    _words = set.intersection(*unprocessed_words)
    _logger.info(
        f"The minimum number of words in word embeddings is: {min_number_of_words_in_all_embeddings} \n \
        The word embedding that has the minimum number of words is {word_emb}"
    )

    return list(_words)


def get_words_with_hyponyms(_words):
    """Fetches hyponyms of words.

    This method makes consecutive calls to an API and returns the list of hyponyms associated with a word.

    Typical usage exampl:
        words_with hyponyms = get_words_with_hyponyms(words)

    Args:
        _words = A list of words whose hyponyms are to be returned

    Returns:
        words_with hyponyms: A dictionary of words with their related hyponyms

    Raises:
        Network Error, All sort of funky stuff. This method does a lot.
    """

    print(
        f"About to start making api calls for hyponyms. Total number of  words = {len(_words)}"
    )
    words_with_hyponyms = {}

    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    async def get_hyponym(session, word):
        """Async method for fetching a list of hyponyms associated with a word."""

        print(f"Fetching hyponyms for specific word which is {word}")

        async with session.get(
            f"http://api.datamuse.com/words?rel_spc={word}"
        ) as response:
            try:
                words_with_hyponyms[word] = await response.json()
                print("Read {0} from {1}".format(response.content_length, word))
            except Exception as e:
                print(f"Something went wrong whilst making the API call", e)
                pass

    async def get_all_hyponyms(words):
        """Async method for fetching hyponyms associated with words."""

        print(
            f"Queueing a list of words to fetch their hyponyms, Number of words is: {len(words)}"
        )

        async with aiohttp.ClientSession() as session:
            tasks = []
            for word in words:
                task = asyncio.ensure_future(get_hyponym(session, word))
                tasks.append(task)
            await asyncio.gather(*tasks, return_exceptions=True)

    words_chunks = list(chunks(_words, 100000))  # chunk words into 100000 portions

    # loop through chunks and fetch hyponyms
    for word_chunks in words_chunks:
        start_time = time.time()
        asyncio.get_event_loop().run_until_complete(get_all_hyponyms(word_chunks))
        duration = time.time() - start_time
        print(f"Words processed {len(_words)} in {duration} seconds")

    return words_with_hyponyms


def save_words_with_hyponyms(_words, file_name):
    """Saves json data to a file.

    Serves as a utility function for saving a dictionary of words with their hyponyms to file
    i.e given words and filename, this function saves the words to a file with the given filename

        Typical usage example:

        save_words_with_hyponyms(words, file_name)

    Args:
        _words: A dictionary of words with their corresponding hyponyms
        file_name: Name of file that would be used to save the dictionary of words

    Returns:
        Returns `True` signifying that the program run successfully

    Raises:
        IOError: An error occurred saving file ... No sure but basically something funky happned when saving file
    """

    with open(f"{DATA_PATH}/{file_name}", "w") as file:
        file.write(json.dumps(_words))  # use `json.loads` to do the reverse
    return True


def main(script_args, _logger=None):
    """Main function that prepares train data.

    This function calls other functions and generates train data for examining distributed word embeddings

        Typical usage example:

        main(script_args)

    Args:
        script_args: argparse variables obtained at the time of running script
        _logger: A python logger object for logging run data

    Returns:
        function returns nothings hence returns implicit None
    """
    if not _logger:
        _logger = get_logger(run_id=UUID)

    if not script_args.run_id:
        script_args.run_id = UUID

    file_name = ""  # declared to ensure file_name is always set

    # check if data_source is embeddings or common_words
    if script_args.data_source == "embeddings":

        # This is done to allow for reading large files.
        # Honestly I've forgotten why I did this, but just know it's important :)
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

        words_from_embeddings = (
            f"{DATA_PATH}/{script_args.run_id}_words_from_embeddings.csv"
        )
        if not Path(words_from_embeddings).is_file():
            words = get_words_from_embeddings(_logger)
            with open(words_from_embeddings, "w", encoding="utf-8") as result_file:
                wr = csv.writer(result_file, dialect="excel")
                wr.writerows([words])
        else:
            with open(words_from_embeddings, "r", encoding="utf-8") as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONE)
                words = list(reader)
            words = words[0]
        _logger.info(
            f"Number of common words found in all embeddings dataset is {len(words)}"
        )
        assert (
            len(words)
            <= 400_000  # number is hard coded since The minimum number of words in all word embeddings is 400_000
        ), "The minimum number of words in all word embeddings is 400_000 hence intersection should be less"
        words = re.findall(r"'(\w+)'", str(words))
        file_name = f"{DATA_PATH}/{script_args.run_id}_words_from_word_embeddings_with_hyponyms.txt"
        if not Path(file_name).is_file():
            words_with_hyponyms = get_words_with_hyponyms(words)
            save_words_with_hyponyms(
                words_with_hyponyms,
                f"{script_args.run_id}_words_from_word_embeddings_with_hyponyms.txt",
            )

    elif script_args.data_source == "common_words":

        with open(f"{DATA_PATH}/corncob_lowercase.txt", "r") as reader:
            content = reader.readlines()
            content = [line.strip() for line in content]
        file_name = f"{DATA_PATH}/{script_args.run_id}_words_with_hyponyms.txt"
        if not Path(file_name).is_file():
            words_with_hyponyms = get_words_with_hyponyms(content)
            save_words_with_hyponyms(
                words_with_hyponyms, f"{script_args.run_id}_words_with_hyponyms.txt"
            )

    with open(file_name, "r") as reader:
        words = reader.readlines()
        words = json.loads(words[0])

    _logger.info(
        f"Total number of words with hyponyms obtained after making API calls is {len(words)}."
    )
    words_from_word_embeddings_with_hyponyms = [
        (key, [value["word"] for value in values]) for key, values in words.items()
    ]

    max_hyp = 0
    for data in words_from_word_embeddings_with_hyponyms:
        max_hyp = len(data[1]) if len(data[1]) > max_hyp else max_hyp
        print(f"current biggest number of hypohyms associated with a word is {max_hyp}")

    all_features = []
    for word in words:
        features = [data["word"] for data in words[word]]
        all_features += features
    all_features = set(all_features)

    tmp_file = f"{DATA_PATH}/{script_args.run_id}_{script_args.data_source}_words.csv"
    if not Path(tmp_file).is_file():
        final_words = []
        for word in words_from_word_embeddings_with_hyponyms:
            word = list(word)
            word[1] += [""] * (max_hyp - len(word[1]))
            final_words.append((word[0], word[1]))

        _logger.info(
            f"Total number of hyponyms i.e unique hyponyms is {len(all_features)}"
        )

        with open(tmp_file, "w", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            for tup in final_words:
                writer.writerow([tup[0], *tup[1]])

    train_csv_file = (
        f"{DATA_PATH}/{script_args.run_id}_word_{script_args.data_source}_train.csv"
    )
    if not Path(train_csv_file).is_file():
        df = pd.read_csv(
            tmp_file, header=None
        )  # Add this parameter `low_memory=False` to silence warning :)
        fwe_df = pd.DataFrame(
            0, index=np.arange(len(df)), columns=sorted(list(all_features))
        )
        df.set_index(0, inplace=True)
        fwe_df.insert(loc=0, column="actual_words", value=list(df.index))
        fwe_df.set_index("actual_words", inplace=True)

        for index, row in fwe_df.iterrows():
            for data in df.loc[index].iteritems():
                if row.get(data[1]) is not None:
                    fwe_df.at[index, data[1]] = 1

        fwe_df.to_csv(train_csv_file)
        shape = fwe_df.shape
        _logger.info(
            f"Total number of words and hyponyms is {shape}, \
            where the first value is the number of words and the second the number of unique hyponyms"
        )
    _logger.info("Done preparing train data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds",
        "--data_source",
        type=str,
        default="embeddings",
        help="The source of data to process, it's either `embeddings` or `common_words`",
    )
    parser.add_argument(
        "-id",
        "--run_id",
        type=str,
        default=None,
        help="Provide a unique identifier which would be used to track the running of the experiment,\
             in the case where it's not provided one will be generated for you. \
             In order to continue the experiment from when it failed,provide it's unique identifier",
    )
    args: Namespace = parser.parse_args()
    main(args)
