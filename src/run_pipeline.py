import argparse
from argparse import Namespace
from datetime import datetime

from .prepare_train import main as main_prepare_train
from .process_train import main as main_process_train
from .utils import get_logger, generate_uuid

logger = get_logger()


def main(script_args):
    run_id = generate_uuid()
    logger.info(f"preparing train data, started at: {datetime.now()}")
    main_prepare_train(script_args, logger, run_id)

    logger.info(f"processing train data, started at: {datetime.now()}")
    main_process_train(script_args, logger, run_id)

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
        help="The source of data to process, it's `embeddings`, `common_words` or `all`",
    )
    args: Namespace = parser.parse_args()
    main(args)
