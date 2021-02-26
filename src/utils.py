import base64
import datetime
import logging
import os
import socket
import sys
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from sklearn import metrics
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             roc_auc_score)
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from models import (FASTTEXT_CRAWL_SUB_300D, FASTTEXT_CRAWL_VEC_300D,
                    FASTTEXT_WIKI_SUB_300D, FASTTEXT_WIKI_VEC_300D,
                    GLOVE_6B_50D, GLOVE_6B_100D, GLOVE_6B_200D, GLOVE_6B_300D,
                    GLOVE_42B_300D, GLOVE_840B_300D, GLOVE_TWITTER_27B_25D,
                    GLOVE_TWITTER_27B_50D, GLOVE_TWITTER_27B_100D,
                    GLOVE_TWITTER_27B_200D, WORD2VEC_GOOGLE_NEWS_300D)

from .logs import LOGS_ROOT as LOG_PATH

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
    "GLOVE_6B_50D": (GLOVE_6B_50D, 50, "glove"),
    "GLOVE_6B_100D": (GLOVE_6B_100D, 100, "glove"),
    "GLOVE_6B_200D": (GLOVE_6B_200D, 200, "glove"),
    "GLOVE_6B_300D": (GLOVE_6B_300D, 300, "glove"),
    "GLOVE_42B_300D": (GLOVE_42B_300D, 300, "glove"),
    "GLOVE_840B_300D": (GLOVE_840B_300D, 300, "glove"),
    "GLOVE_TWITTER_27B_25D": (GLOVE_TWITTER_27B_25D, 25, "glove"),
    "GLOVE_TWITTER_27B_50D": (GLOVE_TWITTER_27B_50D, 50, "glove"),
    "GLOVE_TWITTER_27B_100D": (GLOVE_TWITTER_27B_100D, 100, "glove"),
    "GLOVE_TWITTER_27B_200D": (GLOVE_TWITTER_27B_200D, 200, "glove"),
    "WORD2VEC_GOOGLE_NEWS_300D": (WORD2VEC_GOOGLE_NEWS_300D, 300, "word2vec"),
    "FASTTEXT_CRAWL_SUB": (FASTTEXT_CRAWL_SUB_300D, 300, "fasttext"),
    "FASTTEXT_CRAWL_VEC_300D": (FASTTEXT_CRAWL_VEC_300D, 300, "fasttext"),
    "FASTTEXT_WIKI_SUB_300D": (FASTTEXT_WIKI_SUB_300D, 300, "fasttext"),
    "FASTTEXT_WIKI_VEC_300D": (FASTTEXT_WIKI_VEC_300D, 300, "fasttext"),
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

        current_time = datetime.datetime.now().strftime("%Y%b%d_%H-%M-%S")
        comment = "%s-%d%s" % (self.today, self.today_seconds, self.param_string)
        log_dir = os.path.join(
            "runs", current_time + "_" + socket.gethostname() + "_" + args.run_id
        )

        self.writer = SummaryWriter(log_dir=log_dir)
        self.verbose = args.verbose

        with open(f"{log_dir}/metadata.txt", "w") as text_file:
            print(f"Params used to run the experiment: {comment}", file=text_file)

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


def _load_word_embedding_model(file=None, word_embedding_type="glove"):
    model = {}
    if file is None:
        file, *ign = embeddings.get("GLOVE_6B_300D")
    print("Loading Model")
    if word_embedding_type == "glove":
        df = pd.read_csv(file, sep=" ", quoting=3, header=None, index_col=0)
        model = {key: val.values for key, val in df.T.items()}
        print(len(model), " words loaded!")
    elif word_embedding_type == "word2vec":
        model = KeyedVectors.load_word2vec_format(file, binary=True)
    elif word_embedding_type == "fasttext":
        model = KeyedVectors.load_word2vec_format(file, binary=False)
    return model


def _get_word_embeddings(word):
    word_embedding = None
    try:
        word_embedding = WORD_EMBEDDINGS_MODEL.get(word, None)
    except AttributeError as e:
        print(e)
        try:
            word_embedding = WORD_EMBEDDINGS_MODEL[word]
        except Exception as e:
            print(e)
    return word_embedding


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
            if key != "word_embeddings":
                temp_row.append(row[key])
            else:
                g_em = Variable(torch.Tensor(row[key])).to(device)
        temp_x.append(
            torch.cat([Variable(torch.Tensor(temp_row)).to(device), g_em], dim=0)
        )

    return torch.stack(temp_x, 0)


def data_loader(word, file_name):
    df = pd.read_csv(file_name)
    # df = df.drop(["GLOVE.6B"], axis=1, errors="ignore")
    df["word_embeddings"] = df["actual_words"].apply(_get_word_embeddings)
    df.dropna(inplace=True)
    x_data = df.loc[:, df.columns == "word_embeddings"]
    y_data = df.loc[:, df.columns == word]
    y_data = df_to_tensor(y_data)
    x_data = complex_df_to_tensor(x_data)
    return x_data, y_data, list(df.index.values)


class DatasetLoader(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, word, selected_partition, file_name):
        "Initialization"
        x_data, y_data, indexes = data_loader(word, file_name)

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
def get_words(file_name, threshold=100):
    df = pd.read_csv(file_name)
    subset = df[df.columns.difference(["actual_words", "GLOVE.6B"])]
    # remove columns below threshold
    df_with_threshold = subset.loc[:, (subset.sum(axis=0) >= threshold)].copy()
    return df_with_threshold.columns


####################################################### End all words ##########################################################


def get_logger(run_id=None):
    """Returns a generic logger for logging relevant information pertaining to run

    Returns:
        Python logger object
    """
    # create logger with 'spam_application'
    logger = logging.getLogger("ce_application")
    logger.setLevel(logging.DEBUG)

    if not run_id:
        run_id = generate_uuid()

    file_path = f"{LOG_PATH}/{run_id}_log_{datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')}.log"
    if not logger.handlers:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.file_path = file_path

    return logger


def generate_uuid():
    """Generates a uuid 4 string, in this context for tracking each run of the experiment

    Returns:
        an ascii friendly uuid4 string.
    """
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def get_train_run_parser(parser):
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
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=100,
        help="The threshold for filtering columns of hyponyms",
    )
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
    return parser


def get_trained_models(models_filenames):
    """Get a list of trained models by (word, word_embedding, model_name) stored on disk

    This function fecthes a list of models saved in a directory, mind you these models must be saved trained models from this experiment since the function expects a specific naming convention

    Args:
        models_filenames: A list of files which are trained models

    Returns:
        Trained models represenyed as (word, word_embedding, model_name)"""
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
