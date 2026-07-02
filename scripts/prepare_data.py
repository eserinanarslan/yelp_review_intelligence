import argparse
import sys

from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.data_preparation import YelpDataPreparation
from yelp_review_intelligence.exceptions import DataPreparationError
from yelp_review_intelligence.logger import setup_logger


# Configure project logger
logger = setup_logger(__name__)


def parse_args():
    """
    Parse command-line arguments.
    Allows selecting a custom configuration file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    """
    End-to-end data preparation pipeline.

    Steps:
    1. Load project configuration.
    2. Initialize the data preparation pipeline.
    3. Build train and test datasets.
    4. Save processed datasets for model training.
    """
    try:
        # Read command-line arguments
        args = parse_args()

        logger.info("Starting data preparation pipeline.")
        logger.info("Using config file: %s", args.config)

        # Load project configuration
        config = ProjectConfig.from_yaml(args.config)

        # Initialize the data preparation pipeline
        pipeline = YelpDataPreparation(config)

        # Execute the full preprocessing pipeline
        train_df, test_df = pipeline.run()

        # Define output locations
        train_path = config.processed_dir / "train_model_df.pkl"
        test_path = config.processed_dir / "test_model_df.pkl"

        # Save processed datasets for downstream modeling
        train_df.to_pickle(train_path)
        test_df.to_pickle(test_path)

        logger.info(
            "Saved train dataframe to %s with shape %s",
            train_path,
            train_df.shape,
        )
        logger.info(
            "Saved test dataframe to %s with shape %s",
            test_path,
            test_df.shape,
        )

        logger.info("Data preparation completed successfully.")

    except Exception as exc:
        # Log the full exception and raise a project-specific error
        logger.exception("Data preparation failed.")
        raise DataPreparationError(
            "Data preparation pipeline failed."
        ) from exc


if __name__ == "__main__":
    try:
        # Execute the pipeline
        main()
    except DataPreparationError:
        # Exit with a non-zero status code if the pipeline fails
        sys.exit(1)