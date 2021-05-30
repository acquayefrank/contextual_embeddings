import argparse
import os
import csv
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold

import torch
from joblib import Parallel, delayed

import src.utils as utils
from data import DATA_ROOT as DATA_PATH

from .models import LogisticRegression, SingleLayeredNN
from .utils import (
    DatasetLoader,
    _load_word_embedding_model,
    embeddings,
    get_device,
    get_train_run_parser,
    get_words,
    evaluate,
    get_logger,
)
from evaluation import EVALUATION_ROOT

torch.set_num_threads(4)


def train_on_specific_embedding(
    word_embedding,
    total_number_of_word_embeddings,
    script_args,
    words,
    filename,
    cross_val_filenames,
    params,
    max_epochs,
    *emb_meta_data,
):
    print(emb_meta_data)
    file_path, num_features, word_embedding_type = emb_meta_data
    print(f"{word_embedding}: is being processed")
    total_number_of_word_embeddings = total_number_of_word_embeddings - 1
    print(
        f"{total_number_of_word_embeddings} word_embedding(s) left to process after the current word_embedding"
    )

    script_args.word_embeddings = word_embedding

    print(f"{word_embedding} is the word embedding to set \n")
    print(f"{script_args.word_embeddings} is the actual word embedding being used \n")

    print(f"{file_path} \n")
    print(f"{num_features}\n")
    print(f"{word_embedding_type}\n")

    utils.WORD_EMBEDDINGS_MODEL = _load_word_embedding_model(
        file=file_path, word_embedding_type=word_embedding_type
    )

    total_number_of_words = len(words)

    print(f"Total Number of words: {total_number_of_words}")

    for word in words:
        print(f"{word}: is being processed")
        total_number_of_words = total_number_of_words - 1
        print(f"{total_number_of_words} word(s) left to process after the current word")
        script_args.word = word

        try:
            training_set = DatasetLoader(script_args.word, "train", file_name=filename)
        except TypeError as e:
            print(e)
            continue
        # training_set = DatasetLoader(script_args.word, "train")

        training_generator = torch.utils.data.DataLoader(training_set, **params)

        # TODO: Change to CrossEntropy if results are funny but keeping this for now due to
        # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        criterion = torch.nn.BCELoss(reduction="mean")

        if script_args.all_models:
            models = [
                ("LogisticRegression", LogisticRegression(num_features, 1)),
                ("SingleLayeredNN", SingleLayeredNN(num_features, num_features, 1)),
            ]
        else:
            models = [
                ("LogisticRegression", LogisticRegression(num_features, 1)),
            ]
        # implement slider
        kf = KFold(n_splits=10)
        training_generator_list = list(training_generator)
        for index, (train_index, test_index) in enumerate(
            kf.split(training_generator_list)
        ):
            print("TRAIN:", train_index, "TEST:", test_index)
            train_data = [training_generator_list[index] for index in train_index]
            test_data = [training_generator_list[index] for index in test_index]
            print("TRAIN:", len(train_data), "TEST:", len(test_data))

            for model_name, model in models:
                try:
                    if (
                        f"{word}_{word_embedding}_{model_name}_{index}.csv"
                        in cross_val_filenames
                    ):
                        continue
                    else:
                        headers = [
                            "Training Scenario",
                            "Accuracy",
                            "AUC",
                            "Model Name",
                        ]
                        file_name = f"{EVALUATION_ROOT}/{script_args.run_id}_cross_validation/{word}_{word_embedding}_{model_name}_{index}.csv"
                        with open(file_name, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(headers)

                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=script_args.learning_rate
                    )

                    use_cuda = torch.cuda.is_available()
                    device = torch.device("cuda:0" if use_cuda else "cpu")
                    # torch.backends.cudnn.benchmark = True

                    model.to(device)

                    loss_values = []
                    for epoch in range(max_epochs):
                        # Training
                        running_loss = 0.0
                        for i, data in enumerate(train_data, 0):
                            # data is a list of [features, labels]
                            features, labels = data
                            # Transfer to GPU
                            features, labels = features.to(device), labels.to(device)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # Model computations
                            model.train()  # Take this out

                            # forward + backward + optimize
                            predictions = model(features)
                            # Compute Loss
                            loss = criterion(predictions, labels)
                            # Backward pass
                            loss.backward()
                            optimizer.step()

                            running_loss += loss.item() * features.size(0)

                        loss_values.append(running_loss / len(training_generator))
                    x_data = torch.cat([x for x, _ in test_data])
                    y_data = torch.cat([y for _, y in test_data])
                    scores = evaluate(model, x_data, y_data)

                    results = [word, scores["_accuracy"], scores["_auc"], model_name]
                    with open(file_name, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(results)
                except RuntimeError as e:
                    print(e)
                    continue


def main(script_args):
    cross_validation_results: str = f"{EVALUATION_ROOT}/{script_args.run_id}_cross_validation"
    try:
        os.mkdir(cross_validation_results)
        print("Directory ", cross_validation_results, " Created ")
    except FileExistsError:
        print("Directory ", cross_validation_results, " already exists")

    _, _, cross_val_filenames = next(os.walk(cross_validation_results))

    logger = get_logger(run_id=script_args.run_id)
    logger.info(f"about to start cross-validation: {datetime.now()}")
    logger.info(f"Device: {get_device()}")
    logger.info(f"Pytorch Version: {torch.__version__}")
    logger.info(f"{script_args.__repr__()}")

    params = {"batch_size": script_args.batch_size, "shuffle": True, "num_workers": 0}
    max_epochs = script_args.epoch_num

    filename = f"{DATA_PATH}/{script_args.run_id}_{script_args.data_source}_{script_args.threshold}_final_data.csv"
    # Generators
    if script_args.all:
        words = get_words(file_name=filename)
    else:
        words = [
            script_args.word,
        ]
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

    total_number_of_word_embeddings = len(word_embeddings)
    logger.info(
        f"Description; Number of word embeddings being trained:  {total_number_of_word_embeddings}"
    )

    # use a sequential backend if you encounter strange errors due to some kind of race condition whilst training models
    # Run each embedding as it's own thread
    if not Path(f"{cross_validation_results}/{script_args.run_id}.lock").is_file():
        print("lock file not found hence processing")
        _ = Parallel(n_jobs=8, backend="sequential", verbose=5, require=None)(
            delayed(train_on_specific_embedding)(
                word_embedding,
                total_number_of_word_embeddings,
                script_args,
                words,
                filename,
                cross_val_filenames,
                params,
                max_epochs,
                *embeddings.get(word_embedding),
            )
            for word_embedding in word_embeddings
        )
        open(f"{cross_validation_results}/{script_args.run_id}.lock", "w").close()


if __name__ == "__main__":
    # Use arguments passing to command line in Linux OS
    parser = argparse.ArgumentParser()
    parser = get_train_run_parser(parser)
    args = parser.parse_args()
    main(args)
