import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from data import FINAL_DATA
from models import (
    GLOVE_6B_50D,
    GLOVE_6B_100D,
    GLOVE_6B_200D,
    GLOVE_6B_300D,
    GLOVE_42B_300D,
)


fig_size = (15, 10)
sns.set(
    rc={
        "figure.figsize": fig_size,
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
    },
    style="white",  # nicer layout
)


embeddings = {
    "GLOVE_6B_50D": (GLOVE_6B_50D, 50),
    "GLOVE_6B_100D": (GLOVE_6B_100D, 100),
    "GLOVE_6B_200D": (GLOVE_6B_200D, 200),
    "GLOVE_6B_300D": (GLOVE_6B_300D, 300),
    "GLOVE_42B_300D": (GLOVE_42B_300D, 300),
}


# This is set to None to prevent wierd erros
WORD_EMBEDDINGS_MODEL = {}


class Logger:
    """
    Simple class for logging.
    """

    def __init__(self, args):
        self.start_time = datetime.datetime.now()
        self.timer_start = self.start_time
        self.today = datetime.date.today()
        self.today_seconds = (
            self.start_time
            - self.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        ).total_seconds()

        self.param_string = self.params_help(args)

        self.writer = SummaryWriter(
            comment="%s-%d%s" % (self.today, self.today_seconds, self.param_string,)
        )
        self.verbose = args.verbose

    def help(self, verbose=0):
        if verbose < 100:
            return "Logging started. Current date and time: {}.".format(self.start_time)
        return ""

    def __repr__(self):
        # Print parameters:
        return self.help(self.verbose)

    def params_help(self, args):
        string = ""
        for arg in vars(args):
            string += "_{}-{}".format(arg, getattr(args, arg))
        return string

    def set_timer(self):
        self.timer_start = datetime.datetime.now()

    def print_time(self, str=""):
        str = "of " + str + " "
        time_str = "Time elapsed {}since timer reset: {} (sec).".format(
            str, round((datetime.datetime.now() - self.timer_start).total_seconds(), 2)
        )
        if self.verbose < 100:
            print(time_str)
        self.writer.add_text("Description", time_str)

        time_str = "Total time elapsed since start: {} (sec).".format(
            round((datetime.datetime.now() - self.start_time).total_seconds(), 2)
        )
        if self.verbose < 100:
            print(time_str)
        self.writer.add_text("Description", time_str)


def _load_glove_model(File=None):
    if File is None:
        File, _ = embeddings.get("GLOVE_6B_300D")
    print("Loading Glove Model")
    gloveModel = {}
    with open(File, "r", encoding="utf8") as f:
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel


def _get_word_embeddings(word):
    return WORD_EMBEDDINGS_MODEL.get(word, None)


def get_device():
    # determine the supported device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")  # don't have GPU
    return device


################################# Load Data ###############################################################
# convert a df to tensor to be used in pytorch
def df_to_tensor(s_df):
    device = get_device()
    return Variable(torch.Tensor(s_df.values)).to(device)


def complex_df_to_tensor(_df):
    device = get_device()
    temp_x = []
    for index, row in _df.iterrows():
        temp_row = []
        row = row.to_dict()
        g_em = []
        for key in row.keys():
            if key != "GLOVE.6B":
                temp_row.append(row[key])
            else:
                g_em = Variable(torch.Tensor(row[key])).to(device)
        temp_x.append(
            torch.cat([Variable(torch.Tensor(temp_row)).to(device), g_em], dim=0)
        )

    return torch.stack(temp_x, 0)


def data_loader(word):
    df = pd.read_csv(FINAL_DATA)
    df["GLOVE.6B"] = df["actual_words"].apply(_get_word_embeddings)
    x_data = df.loc[:, df.columns == "GLOVE.6B"]
    y_data = df.loc[:, df.columns == word]
    y_data = df_to_tensor(y_data)
    x_data = complex_df_to_tensor(x_data)
    return x_data, y_data, list(df.index.values)


class DatasetLoader(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, word, selected_partition):
        "Initialization"
        x_data, y_data, indexes = data_loader(word)

        # Datasets
        partition = {"train": indexes, "validation": indexes}
        self.list_IDs = partition[selected_partition]

        self.labels = dict(zip(indexes, y_data))

        self.features = dict(zip(indexes, x_data))

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.features[ID]
        y = self.labels[ID]

        return X, y


################################# End Load Data ###############################################################


#################################### Plots #####################################################################
def plot_learning_rate(train_loss):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_loss)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Learning rate")
    plt.title("Learning Curve")
    return fig


def plot_tpr(fpr, tpr, label):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, linestyle="--", label=label)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("True Positive Rate vs False Positive Rate")
    return fig


def plot_precision_recall(recall, precision, label):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, marker=".", label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Recall vs Precision")
    return fig


# helper functions
def features_to_probs(model, features):
    """
    Generates predictions and corresponding probabilities from a trained
    """
    output = model(features)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().data.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(model, features, labels):
    preds, probs = features_to_probs(model, features)
    # plot the features in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        ax.set_title("Learning Curve")
    return fig


################################### End Plots ####################################################################


def evaluate(model, x_data, y_data):
    y_pred = model(x_data)
    y_data = y_data.cpu().detach().numpy()
    x_data = x_data.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    y_pred_labels = []
    for p in y_pred:
        if p < 0.5:
            label = 0
        else:
            label = 1
        y_pred_labels.append(label)

    _accuracy = accuracy_score(y_data, y_pred_labels)
    _f1_score = f1_score(y_data, y_pred_labels, average="micro")
    _roc_auc_score = roc_auc_score(y_data, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_data, y_pred, pos_label=1)
    _auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(y_data, y_pred)

    return {
        "_accuracy": _accuracy,
        "_f1_score": _f1_score,
        "_roc_auc_score": _roc_auc_score,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "_auc": _auc,
        "precision": precision,
        "recall": recall,
    }


####################################################### Get all words ##########################################################
def get_words(threshold=100):
    df = pd.read_csv(FINAL_DATA)
    subset = df[df.columns.difference(["actual_words", "GLOVE.6B"])]
    # remove columns below threshold
    df_with_threshold = subset.loc[:, (subset.sum(axis=0) >= threshold)].copy()
    return df_with_threshold.columns


####################################################### End all words ##########################################################
