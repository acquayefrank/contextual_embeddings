import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from .prepare_train import main as main_prepare_train
from .process_train import main as main_process_train
from .run_valid_test import main as main_valid_test
from .train import main as main_train
from .utils import generate_uuid, get_logger, get_train_run_parser
from .visualize_models import main as main_visualize_models

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
    print(f"UUID for run is: {script_args.run_id}")
    logger.info(f"preparing train data, started at: {datetime.now()}")
    main_prepare_train(script_args, logger)

    print(f"UUID for run is: {script_args.run_id}")
    logger.info(f"processing train data, started at: {datetime.now()}")
    main_process_train(script_args, logger)

    if not Path(f"./runs/{script_args.run_id}.lock").is_file():
        print(f"UUID for run is: {script_args.run_id}")
        logger.info(f"training, started at: {datetime.now()}")
        main_train(script_args)
    else:
        print("models already trained")

    print(f"UUID for run is: {script_args.run_id}")
    main_valid_test(script_args)
    print(f"UUID for run is: {script_args.run_id}")
    main_visualize_models(script_args)
    print(
        f"processing is done, copy and share your results. UUID for run is: {script_args.run_id}"
    )
    print(
        "perhaps in future I'll zip stuff for you for now copy and share your results :) Good Bye."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = get_train_run_parser(parser)
    args: Namespace = parser.parse_args()
    main(args)
