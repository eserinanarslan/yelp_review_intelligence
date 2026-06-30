import argparse
import sys

import pandas as pd

from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.exceptions import ModelTrainingError
from yelp_review_intelligence.logger import setup_logger
from yelp_review_intelligence.modeling import SentimentModelTrainer
from yelp_review_intelligence.utils import save_json, save_model


logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    """
    Train and evaluate the sentiment model.
    """
    try:
        args = parse_args()
        logger.info("Starting model training pipeline.")
        logger.info("Using config file: %s", args.config)

        config = ProjectConfig.from_yaml(args.config)

        train_path = config.processed_dir / "train_model_df.pkl"
        test_path = config.processed_dir / "test_model_df.pkl"

        logger.info("Loading train data from %s", train_path)
        logger.info("Loading test data from %s", test_path)

        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)

        trainer = SentimentModelTrainer(config)

        logger.info("Running ablation experiments.")
        results_df, _ = trainer.run_ablation(train_df, test_df)

        ablation_path = config.processed_dir / "ablation_results.csv"
        results_df.to_csv(ablation_path, index=False)
        logger.info("Saved ablation results to %s", ablation_path)

        best_experiment_name = results_df.loc[0, "model_name"]
        logger.info("Best experiment selected: %s", best_experiment_name)

        logger.info("Running grid search for the best experiment.")
        best_model, test_output, metrics = trainer.tune_best_model(
            train_df,
            test_df,
            best_experiment_name,
        )

        labels = config.raw["model"]["labels"]
        cm_df, cm_norm_df = trainer.confusion_matrix_df(
            test_output["target_sentiment"],
            test_output["predicted_sentiment"].to_numpy(),
            labels,
        )

        importance_tables = trainer.feature_importance(best_model)

        save_model(best_model, config.model_dir / "sentiment_model.pkl")
        test_output.to_pickle(config.processed_dir / "test_predictions.pkl")

        cm_df.to_csv(config.output_dir / "confusion_matrix.csv")
        cm_norm_df.to_csv(config.output_dir / "confusion_matrix_normalized.csv")

        for class_name, table in importance_tables.items():
            table.to_csv(config.output_dir / f"feature_importance_{class_name}.csv", index=False)

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

        save_json(
            metrics["classification_report"],
            config.output_dir / "classification_report.json",
        )

        logger.info("Model training completed successfully.")
        logger.info("Best experiment: %s", best_experiment_name)
        logger.info("Test accuracy: %.4f", metrics["accuracy"])
        logger.info("Test macro F1: %.4f", metrics["macro_f1"])

    except Exception as exc:
        logger.exception("Model training failed.")
        raise ModelTrainingError("Model training pipeline failed.") from exc


if __name__ == "__main__":
    try:
        main()
    except ModelTrainingError:
        sys.exit(1)