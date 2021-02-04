import argparse
import os
from os.path import isfile, join
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import torch
from joblib import Parallel, delayed

import src.utils as utils
from data import DATA_ROOT as DATA_PATH
from models import MODELS_PATH

from .models import LogisticRegression, SingleLayeredNN
from .utils import (
    DatasetLoader,
    Logger,
    _load_word_embedding_model,
    data_loader,
    embeddings,
    evaluate,
    get_device,
    get_words,
    plot_learning_rate,
    plot_precision_recall,
    plot_tpr,
    get_train_run_parser,
)


def train_on_specific_embedding(
    word_embedding,
    total_number_of_word_embeddings,
    script_args,
    writer,
    words,
    filename,
    trained_models_root,
    params,
    max_epochs,
    trained_models,
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
    writer.add_text(
        "Description", f"Number of words being trained: {total_number_of_words}"
    )
    print(f"Total Number of words: {total_number_of_words}")

    for word in words:
        print(f"{word}: is being processed")
        total_number_of_words = total_number_of_words - 1
        print(f"{total_number_of_words} word(s) left to process after the current word")
        script_args.word = word
        writer.add_text(
            "Description",
            f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}",
        )
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

        for model_name, model in models:
            try:
                if (word, word_embedding, model_name) in trained_models:
                    print(num_features, file_path, word_embedding, model_name, model)
                    continue
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

                    for i, data in enumerate(training_generator, 0):
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
                    writer.add_scalar(
                        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings} training loss, Model being used: {model}",
                        running_loss / len(training_generator),
                        epoch,
                    )
                writer.add_figure("Learning Rate", plot_learning_rate(loss_values))
                model_path = f"{trained_models_root}/{script_args.word}_{word_embedding}_{model_name}.pth"
                torch.save(model.state_dict(), model_path)
            except RuntimeError as e:
                print(e)
                print(num_features, file_path, word_embedding)
                continue


def get_trained_models(models_filenames):
    trained_models = []
    for m_f in models_filenames:
        meta_data = m_f.split("_")
        word = meta_data[0]
        model = meta_data[-1][:-4]
        embedding = "_".join(meta_data[1:-1])
        if word == "POS":
            word = "_".join(meta_data[:3])
            embedding = "_".join(meta_data[3:-1])
        print(word, embedding, model)
        trained_models.append((word, embedding, model))
    return trained_models


def run_test_on_models(script_args, filename, writer, model):
    # Load
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    # Tests and Accuracies
    x_data, y_data, _ = data_loader(script_args.word, filename)

    writer.add_graph(model, x_data)

    scores = evaluate(model, x_data, y_data)

    writer.add_text(
        "Current shape of data being processed",
        str(f"x_data: {x_data.shape} and y_data: {y_data.shape}"),
    )

    writer.add_text(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, Accuracy",
        str(scores["_accuracy"]),
    )
    writer.add_text(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, F1 Score",
        str(scores["_f1_score"]),
    )
    writer.add_text(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, roc_auc_score",
        str(scores["_roc_auc_score"]),
    )
    writer.add_text(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, fpr",
        str(scores["fpr"]),
    )
    writer.add_text(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, tpr",
        str(scores["tpr"]),
    )
    writer.add_text(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, average thresholds",
        str(scores["thresholds"]),
    )
    writer.add_text(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, _auc",
        str(scores["_auc"]),
    )
    writer.add_text(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, Precision",
        str(scores["precision"]),
    )
    writer.add_text(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, Recall",
        str(scores["recall"]),
    )

    writer.add_figure(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, False Positive Rate",
        plot_tpr(scores["fpr"], scores["tpr"], script_args.word),
    )
    writer.add_figure(
        f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings},\
                     Model being used: {model}, Precision vs Recall",
        plot_precision_recall(scores["recall"], scores["precision"], script_args.word),
    )


def get_actual_models(trained_models_root, model_type):
    model_paths = [
        f
        for f in os.listdir(trained_models_root)
        if isfile(join(trained_models_root, f))
    ]
    all_models = [
        model for model in model_paths if model.split("_")[-1][:-4] == model_type
    ]
    models = []
    for index, mdl in enumerate(all_models):
        emb_name = "_".join(mdl.split("_")[1:-1])
        if embeddings.get(emb_name):
            embeddings_data = embeddings.get(emb_name)
        else:
            embeddings_data = embeddings.get("_".join(mdl.split("_")[3:-1]))
        file_path, dim, embedding_type = embeddings_data
        if model_type == "LogisticRegression":
            models.append(LogisticRegression(dim, 1))
        elif model_type == "SingleLayeredNN":
            models.append(SingleLayeredNN(dim, dim, 1))
    return models


def main(script_args):
    trained_models_root: str = f"{MODELS_PATH}/{script_args.run_id}_trained_models"
    try:
        os.mkdir(trained_models_root)
        print("Directory ", trained_models_root, " Created ")
    except FileExistsError:
        print("Directory ", trained_models_root, " already exists")

    _, _, models_filenames = next(os.walk(trained_models_root))
    trained_models = get_trained_models(models_filenames)

    pytorch_device = f"Device: {get_device()}"
    pytorch_version = f"Pytorch Version: {torch.__version__}"
    print(pytorch_version)
    print(pytorch_device)
    logger = Logger(script_args)  # Create a logger.
    # Print information about the logger to the screen.
    # The amount of the information depends on the verbosity level specified by the user in script_args.
    print(logger)

    # Print information to tensorboard about the parameters of the training and the model
    writer = logger.writer
    writer.add_text("Description", logger.help())

    writer.add_text("Description", pytorch_version)
    writer.add_text("Description", pytorch_device)

    # Print script_args to the screen
    if script_args.verbose < 100:
        print(script_args)
    # Print the information about arguments to tensorboard
    writer.add_text("Description", script_args.__repr__())

    writer.add_text("Description", logger.param_string)

    logger.set_timer()

    # Parameters
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
    writer.add_text(
        "Description",
        f"Number of word embeddings being trained: {total_number_of_word_embeddings}",
    )
    print(f"Total Number of word embeddings: {total_number_of_word_embeddings}")

    # use a sequential backend if you encounter strange errors due to some kind of race condition whilst training models
    _ = Parallel(n_jobs=-1, backend="threading", verbose=5, require=None)(
        delayed(train_on_specific_embedding)(
            word_embedding,
            total_number_of_word_embeddings,
            script_args,
            writer,
            words,
            filename,
            trained_models_root,
            params,
            max_epochs,
            trained_models,
            *embeddings.get(word_embedding),
        )
        for word_embedding in word_embeddings
    )

    # Tests for LogisticRegression
    _ = Parallel(n_jobs=-1, backend="threading", verbose=5)(
        delayed(run_test_on_models)(script_args, filename, writer, model)
        for model in get_actual_models(
            trained_models_root, model_type="LogisticRegression"
        )
    )

    # Tests for SingleLayeredNN
    _ = Parallel(n_jobs=-1, backend="threading", verbose=5)(
        delayed(run_test_on_models)(script_args, filename, writer, model)
        for model in get_actual_models(
            trained_models_root, model_type="SingleLayeredNN"
        )
    )

    logger.print_time()
    writer.close()
    open(f"{script_args.run_id}.lock", "w").close()
    print("Done. Bye.")


if __name__ == "__main__":
    # Use arguments passing to command line in Linux OS
    parser = argparse.ArgumentParser()
    parser = get_train_run_parser(parser)
    args = parser.parse_args()
    main(args)
