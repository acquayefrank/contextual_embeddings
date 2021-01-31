import argparse
from argparse import Namespace
from datetime import datetime

from .prepare_train import main as main_prepare_train
from .process_train import main as main_process_train
from .utils import get_logger, generate_uuid

UUID = generate_uuid()


def main(script_args):
    """Main method that runs the entire experiment.

    This method serves as a pipeline for running the entire codebase.
    It first runs the main method for preparing the train data.
    It then runs the main method for processing the train data.
    It trains the models and saves them.

    Known issues:
        Since the code was written to be modular, each script can be run individually,
        in cases you choose that route use the same run_id in order not to run into funny issues

    Args:
        script_args: This is an argparse object containing all parameters parsed at the terminal


    """
    if not script_args.run_id:
        script_args.run_id = UUID
    logger = get_logger(run_id=script_args.run_id)
    print(f"UUID for run is: {UUID}")
    logger.info(f"preparing train data, started at: {datetime.now()}")
    main_prepare_train(script_args, logger)

    print(f"UUID for run is: {UUID}")
    logger.info(f"processing train data, started at: {datetime.now()}")
    main_process_train(script_args, logger)

    print(f"UUID for run is: {UUID}")
    print("yet to train")
    print("yet to generate report")
    print("yet to visualize models")


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
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=100,
        help="The threshold for filtering columns of hyponyms",
    )
    args: Namespace = parser.parse_args()
    main(args)
