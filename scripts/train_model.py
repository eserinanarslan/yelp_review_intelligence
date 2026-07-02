import argparse
import sys

import pandas as pd

from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.exceptions import ModelTrainingError
from yelp_review_intelligence.logger import setup_logger
from yelp_review_intelligence.modeling import SentimentModelTrainer
from yelp_review_intelligence.utils import save_json, save_model


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
    End-to-end model training pipeline.

    Steps:
    1. Load processed train and test datasets.
    2. Run ablation experiments to compare feature sets.
    3. Select the best experiment.
    4. Perform hyperparameter tuning using Grid Search.
    5. Evaluate the final model.
    6. Save the trained model and evaluation artifacts.
    """
    try:
        # Read command-line arguments
        args = parse_args()

        logger.info("Starting model training pipeline.")
        logger.info("Using config file: %s", args.config)

        # Load project configuration
        config = ProjectConfig.from_yaml(args.config)

        # Define processed dataset locations
        train_path = config.processed_dir / "train_model_df.pkl"
        test_path = config.processed_dir / "test_model_df.pkl"

        logger.info("Loading train data from %s", train_path)
        logger.info("Loading test data from %s", test_path)

        # Load prepared datasets
        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)

        # Initialize the model trainer
        trainer = SentimentModelTrainer(config)

        # Compare multiple feature configurations (ablation study)
        logger.info("Running ablation experiments.")
        results_df, _ = trainer.run_ablation(train_df, test_df)

        # Save ablation results
        ablation_path = config.processed_dir / "ablation_results.csv"
        results_df.to_csv(ablation_path, index=False)
        logger.info("Saved ablation results to %s", ablation_path)

        # Select the best-performing experiment
        best_experiment_name = results_df.loc[0, "model_name"]
        logger.info("Best experiment selected: %s", best_experiment_name)

        # Optimize the selected model using Grid Search
        logger.info("Running grid search for the best experiment.")
        best_model, test_output, metrics = trainer.tune_best_model(
            train_df,
            test_df,
            best_experiment_name,
        )

        # Generate confusion matrices
        labels = config.raw["model"]["labels"]
        cm_df, cm_norm_df = trainer.confusion_matrix_df(
            test_output["target_sentiment"],
            test_output["predicted_sentiment"].to_numpy(),
            labels,
        )

        # Extract feature importance (when supported by the model)
        importance_tables = trainer.feature_importance(best_model)

        # Save the trained production model
        save_model(best_model, config.model_dir / "sentiment_model.pkl")

        # Save prediction outputs for downstream business intelligence
        test_output.to_pickle(
            config.processed_dir / "test_predictions.pkl"
        )

        # Save evaluation artifacts
        cm_df.to_csv(config.output_dir / "confusion_matrix.csv")
        cm_norm_df.to_csv(
            config.output_dir / "confusion_matrix_normalized.csv"
        )

        # Save feature importance reports
        for class_name, table in importance_tables.items():
            table.to_csv(
                config.output_dir / f"feature_importance_{class_name}.csv",
                index=False,
            )

        # Save model performance metrics
        save_json(
            {
                "best_experiment_name": best_experiment_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "best_params": metrics["best_params"],
                "best_cv_score": metrics["best_cv_score"],
                "feature_cols": metrics["feature_cols"],
            },
            config.output_dir / "model_metrics.json",
        )

        # Save the detailed classification report
        save_json(
            metrics["classification_report"],
            config.output_dir / "classification_report.json",
        )

        logger.info("Model training completed successfully.")
        logger.info("Best experiment: %s", best_experiment_name)
        logger.info("Test accuracy: %.4f", metrics["accuracy"])
        logger.info("Test macro F1: %.4f", metrics["macro_f1"])

    except Exception as exc:
        # Log the full exception and raise a project-specific error
        logger.exception("Model training failed.")
        raise ModelTrainingError(
            "Model training pipeline failed."
        ) from exc


if __name__ == "__main__":
    try:
        # Execute the training pipeline
        main()
    except ModelTrainingError:
        # Exit with a non-zero status code if the pipeline fails
        sys.exit(1)