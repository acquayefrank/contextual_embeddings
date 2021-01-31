import argparse

import torch

import src.utils as utils
from models import TRAINED_MODELS_ROOT

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
)


def main(script_args):

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

    # Generators
    if script_args.all:
        words = get_words()
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

    for word_embedding in word_embeddings:
        print(f"{word_embedding}: is being processed")
        total_number_of_word_embeddings = total_number_of_word_embeddings - 1
        print(
            f"{total_number_of_word_embeddings} word_embedding(s) left to process after the current word_embedding"
        )

        script_args.word_embeddings = word_embedding
        file_path, num_features, word_embedding_type = embeddings.get(
            script_args.word_embeddings
        )
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
            print(
                f"{total_number_of_words} word(s) left to process after the current word"
            )
            script_args.word = word
            writer.add_text(
                "Description",
                f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}",
            )
            try:
                training_set = DatasetLoader(script_args.word, "train")
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
                model_path = (
                    TRAINED_MODELS_ROOT
                    / f"{script_args.word}_{word_embedding}_{model_name}.pth"
                )
                torch.save(model.state_dict(), model_path)
                # Tests and Accuracies
                x_data, y_data, _ = data_loader(script_args.word)

                writer.add_graph(model, x_data)

                writer.add_figure("Learning Rate", plot_learning_rate(loss_values))

                scores = evaluate(model, x_data, y_data)

                writer.add_text(
                    "Current shape of data being processed",
                    str(f"x_data: {x_data.shape} and y_data: {y_data.shape}"),
                )

                writer.add_text(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, Accuracy",
                    str(scores["_accuracy"]),
                )
                writer.add_text(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, F1 Score",
                    str(scores["_f1_score"]),
                )
                writer.add_text(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, roc_auc_score",
                    str(scores["_roc_auc_score"]),
                )
                writer.add_text(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, fpr",
                    str(scores["fpr"]),
                )
                writer.add_text(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, tpr",
                    str(scores["tpr"]),
                )
                writer.add_text(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, average thresholds",
                    str(scores["thresholds"]),
                )
                writer.add_text(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, _auc",
                    str(scores["_auc"]),
                )
                writer.add_text(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, Precision",
                    str(scores["precision"]),
                )
                writer.add_text(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, Recall",
                    str(scores["recall"]),
                )

                writer.add_figure(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, False Positive Rate",
                    plot_tpr(scores["fpr"], scores["tpr"], script_args.word),
                )
                writer.add_figure(
                    f"Word being trained: {script_args.word}, Word Embeddings being used: {script_args.word_embeddings}, Model being used: {model}, Precision vs Recall",
                    plot_precision_recall(
                        scores["recall"], scores["precision"], script_args.word
                    ),
                )

    logger.print_time()
    writer.close()
    print("Done. Bye.")


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


if __name__ == "__main__":
    # Use arguments passing to command line in Linux OS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epoch_num",
        type=int,
        default="5000",
        help="Number of epochs run. Default: 5000",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default="70",
        help="Controls the information shown during training. Default: 70",
    )
    parser.add_argument(
        "-w",
        "--word",
        type=str,
        default="move",
        help="The word you would want to train for. Default: move",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=100,
        help="Controls the batch size. Default: 100",
    ),
    parser.add_argument(
        "-a",
        "--all",
        type=boolean_string,
        default=True,
        help="If set to true, training is done on entire dataset. Default: True",
    )
    parser.add_argument(
        "-ae",
        "--all_embeddings",
        type=boolean_string,
        default=False,
        help="If set to true, training is done on all word embeddings. Default: False",
    )
    parser.add_argument(
        "-am",
        "--all_models",
        type=boolean_string,
        default=False,
        help="If set to true, training is done on all models. Default: False",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Controls learning rate. Default: 0.0001",
    )
    parser.add_argument(
        "-we",
        "--word_embeddings",
        type=str,
        default="GLOVE_6B_300D",
        help="Controls which word embedding should be used. Default: GLOVE_6B_300D",
    )
    args = parser.parse_args()
    main(args)
