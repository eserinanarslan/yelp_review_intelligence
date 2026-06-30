import argparse
import pandas as pd
from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.modeling import SentimentModelTrainer
from yelp_review_intelligence.utils import save_model, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    train_df = pd.read_pickle(config.processed_dir / "train_model_df.pkl")
    test_df = pd.read_pickle(config.processed_dir / "test_model_df.pkl")
    trainer = SentimentModelTrainer(config)

    results_df, _ = trainer.run_ablation(train_df, test_df)
    results_df.to_csv(config.processed_dir / "ablation_results.csv", index=False)
    best_experiment_name = results_df.loc[0, "model_name"]

    best_model, test_output, metrics = trainer.tune_best_model(train_df, test_df, best_experiment_name)
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

    save_json({
        "best_experiment_name": best_experiment_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "best_params": metrics["best_params"],
        "best_cv_score": metrics["best_cv_score"],
        "feature_cols": metrics["feature_cols"],
    }, config.output_dir / "model_metrics.json")
    save_json(metrics["classification_report"], config.output_dir / "classification_report.json")

    print(results_df)
    print(f"Best experiment: {best_experiment_name}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
