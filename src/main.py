import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from .models import LogisticRegression
import src.utils as utils
from .utils import (Logger, evaluate, get_device,
                    plot_learning_rate, plot_precision_recall, plot_tpr, DatasetLoader, 
                    data_loader, plot_classes_preds, _load_glove_model, embeddings, get_words)


def main(args):

    pytorch_device = f"Device: {get_device()}"
    pytorch_version = f"Pytorch Version: {torch.__version__}"
    print(pytorch_version)
    print(pytorch_device)
    logger = Logger(args)  # Create a logger.
    # Print infomration about the logger to the screen.
    # The amount of the information depends on the verbosity level specified by the user in args.
    print(logger)

    # Print information to tensorboard about the parameters of the training and the model
    writer = logger.writer
    writer.add_text("Description", logger.help())

    writer.add_text("Description", pytorch_version)
    writer.add_text("Description", pytorch_device)

    # Print args to the screen
    if args.verbose < 100:
        print(args)
    # Print the information about arguments to tensorboard
    writer.add_text("Description", args.__repr__())

    writer.add_text("Description", logger.param_string)

    logger.set_timer()

    # Parameters
    params = {'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 0}
    max_epochs = args.epoch_num


    # Generators
    if args.all:
        words = get_words()
    else:
        words = [args.word, ]

    file_path, num_features = embeddings.get(args.word_embeddings)
    utils.WORD_EMBEDDINGS_MODEL = _load_glove_model(File=file_path)

    writer.add_text("Description", f"Number of words being trained: {len(words)}")
    total_number_of_words = len(words)
    print(f"Total Number of words: {total_number_of_words}")
    for word in words:
        print(f"{word}: is being processed")
        total_number_of_words = total_number_of_words - 1
        print(f"{total_number_of_words} word(s) left to process after the current word")
        args.word = word
        writer.add_text("Description", f"Word being trained: {args.word}, Word Embeddings being used: {args.word_embeddings}")

        training_set = DatasetLoader(args.word, "train")
        training_generator = torch.utils.data.DataLoader(training_set, **params)

        # currently not in use
        # validation_set = DatasetLoader(args.word, "validation")
        # validation_generator = torch.utils.data.DataLoader(validation_set, **params)

        criterion = torch.nn.BCELoss(reduction="mean")

        model = LogisticRegression(num_features, 1)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


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
                model.train() # Take this out
                
                # forward + backward + optimize
                predictions = model(features)
                # Compute Loss
                loss = criterion(predictions, labels)
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * features.size(0)

            loss_values.append(running_loss / len(training_generator))
            writer.add_scalar(f'{word}: training loss', running_loss / len(training_generator), epoch)
                
                # if i % 100 == 99:    # every 100 mini-batches...

                #     # ...log the running loss
                #     writer.add_scalar(f'{word}: training loss',
                #                     running_loss / 1000,
                #                     epoch * len(training_generator) + i)

                #     # ...log a Matplotlib Figure showing the model's predictions on a
                #     # random mini-batch

                #     writer.add_figure(f'{word}: predictions vs. actuals',
                #                 plot_classes_preds(model, features, labels),
                #                 global_step=epoch * len(training_generator) + i)
                #     running_loss = 0.0

            # # Validation
            # with torch.set_grad_enabled(False):
            #     for features, labels in validation_generator:
            #         # Transfer to GPU
            #         features, labels = features.to(device), labels.to(device)

            #         # Model computations
            #         [...]


        # Loop over epochs
        # train_loss = []
        # for epoch in range(max_epochs):
        #     # Training
        #     running_loss = 0.0
        #     for local_batch, local_labels in training_generator:
        #         # Transfer to GPU
        #         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        #         # Model computations
        #         model.train()
        #         # Forward pass
        #         y_pred = model(local_batch)
        #         # Compute Loss
        #         loss = criterion(y_pred, local_labels)
        #         train_loss.append(loss.item())

        #         optimizer.zero_grad()
        #         # Backward pass
        #         loss.backward()
        #         optimizer.step()

        #         running_loss += loss.item() * local_batch.size(0)
        #     epoch_loss = running_loss / len(local_batch)
        #     train_loss.append(epoch_loss)

        

        # Tests and Accuracies
        x_data, y_data, _ = data_loader(args.word)

        writer.add_graph(model, x_data)

        writer.add_figure("Learning Rate", plot_learning_rate(loss_values))

        scores = evaluate(model, x_data, y_data)

        writer.add_text(f"{word} Accuracy", str(scores["_accuracy"]))
        writer.add_text(f"{word} F1 Score", str(scores["_f1_score"]))
        writer.add_text(f"{word} roc_auc_score", str(scores["_roc_auc_score"]))
        writer.add_text(f"{word} fpr", str( scores["fpr"] ))
        writer.add_text(f"{word} tpr", str( scores["tpr"] ))
        writer.add_text(f"{word} average thresholds", str( scores["thresholds"] ))
        writer.add_text(f"{word} _auc", str(scores["_auc"]))
        writer.add_text(f"{word} Precision", str( scores["precision"] ))
        writer.add_text(f"{word} Recall", str( scores["recall"] ))

        writer.add_figure(
            f"{word} False Positive Rate", plot_tpr(scores["fpr"], scores["tpr"], args.word)
        )
        writer.add_figure(
            f"{word} Precision vs Recall",
            plot_precision_recall(scores["recall"], scores["precision"], args.word),
        )

    logger.print_time()
    writer.close()
    print("Done. Bye.")


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
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
        help="The word you would want to train for. Default: move"
    )
    parser.add_argument(
        "-b",
        "--batch_size", 
        type=int, 
        default=100,
        help="Controls the batch size. Default: 100"
    ),
    parser.add_argument(
        "-a",
        "--all",
        type=boolean_string,
        default=True,
        help="If set to true, training is done on entire dataset. Default: True" 
    )
    parser.add_argument(
        "-lr",
        "--learning_rate", 
        type=float,
        default=0.0001, 
        help="Controls learning rate. Default: 0.0001"
    )
    parser.add_argument(
        "-we",
        "--word_embeddings", 
        type=str, 
        default="GLOVE_6B_300D",
        help="Controls which word embedding should be used. Default: GLOVE_6B_300D"
    )
    args = parser.parse_args()
    main(args)
