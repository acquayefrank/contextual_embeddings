import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from .models import LogisticRegression
from .utils import (Logger, evaluate, get_device,
                    plot_learning_rate, plot_precision_recall, plot_tpr, DatasetLoader, data_loader, plot_classes_preds)


def main(args):
    logger = Logger(args)  # Create a logger.
    # Print infomration about the logger to the screen.
    # The amount of the information depends on the verbosity level specified by the user in args.
    print(logger)

    # Print information to tensorboard about the parameters of the training and the model
    writer = logger.writer
    writer.add_text("Description", logger.help())

    # Print args to the screen
    if args.verbose < 100:
        print(args)
    # Print the information about arguments to tensorboard
    writer.add_text("Description", args.__repr__())

    # Create a dependency parser from Allan AI along with a WordNet lemmatizer.
    logger.set_timer()

    # Parameters
    params = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 0}
    max_epochs = args.epoch_num


    # Generators
    training_set = DatasetLoader("move", "train")
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    # currently not in use
    validation_set = DatasetLoader("move", "validation")
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    criterion = torch.nn.BCELoss(reduction="mean")

    model = LogisticRegression(300, 1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model.to(device)

    running_loss = 0.0
    for epoch in range(max_epochs):
        # Training
    
        for i, data in enumerate(training_generator, 0):
            # data is a list of [features, labels]
            features, labels = data
            # Transfer to GPU
            features, labels = features.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Model computations
            model.train()
            
            # forward + backward + optimize
            predictions = model(features)
            # Compute Loss
            loss = criterion(predictions, labels)
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if i % 10 == 9:    # every 10 mini-batches...

                # ...log the running loss
                writer.add_scalar('training loss',
                                running_loss / 1000,
                                epoch * len(training_generator) + i)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch

                writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(model, features, labels),
                            global_step=epoch * len(training_generator) + i)
                running_loss = 0.0

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
    x_data, y_data, _ = data_loader("move")

    writer.add_graph(model, x_data)

    # writer.add_figure("Learning rates", plot_learning_rate(train_loss))

    scores = evaluate(model, x_data, y_data)

    writer.add_text("Accuracy", str(scores["_accuracy"]))
    writer.add_text("F1 Score", str(scores["_f1_score"]))
    writer.add_text("roc_auc_score", str(scores["_roc_auc_score"]))
    writer.add_text("fpr", str( scores["fpr"] ))
    writer.add_text("tpr", str( scores["tpr"] ))
    writer.add_text("average thresholds", str( scores["thresholds"] ))
    writer.add_text("_auc", str(scores["_auc"]))
    writer.add_text("Precision", str( scores["precision"] ))
    writer.add_text("Recall", str( scores["recall"] ))

    writer.add_figure(
        "False Positive Rate", plot_tpr(scores["fpr"], scores["tpr"], "move")
    )
    writer.add_figure(
        "Precision vs Recall",
        plot_precision_recall(scores["recall"], scores["precision"], "move"),
    )

    writer.close()

    print("Done. Bye.")


if __name__ == "__main__":
    print(f"Pytorch Version: {torch.__version__}")
    print(f"Device: {get_device()}")

    # Use arguments passing to command line in Linux OS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epoch_num",
        type=int,
        default="5000",
        help="Number of epochs run. Default: 20",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default="70",
        help="Controls the information shown during training. Default: 70",
    )
    args = parser.parse_args()
    main(args)
