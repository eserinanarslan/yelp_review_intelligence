import argparse
import sys

from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.data_preparation import YelpDataPreparation
from yelp_review_intelligence.exceptions import DataPreparationError
from yelp_review_intelligence.logger import setup_logger


logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    """
    Run the data preparation pipeline.
    """
    try:
        args = parse_args()
        logger.info("Starting data preparation pipeline.")
        logger.info("Using config file: %s", args.config)

        config = ProjectConfig.from_yaml(args.config)
        pipeline = YelpDataPreparation(config)

        train_df, test_df = pipeline.run()

        train_path = config.processed_dir / "train_model_df.pkl"
        test_path = config.processed_dir / "test_model_df.pkl"

        train_df.to_pickle(train_path)
        test_df.to_pickle(test_path)

        logger.info("Saved train dataframe to %s with shape %s", train_path, train_df.shape)
        logger.info("Saved test dataframe to %s with shape %s", test_path, test_df.shape)
        logger.info("Data preparation completed successfully.")

    except Exception as exc:
        logger.exception("Data preparation failed.")
        raise DataPreparationError("Data preparation pipeline failed.") from exc


if __name__ == "__main__":
    try:
        main()
    except DataPreparationError:
        sys.exit(1)