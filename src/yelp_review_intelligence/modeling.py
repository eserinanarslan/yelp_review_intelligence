from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import ProjectConfig


@dataclass
class SentimentModelTrainer:
    config: ProjectConfig

    @staticmethod
    def stratified_sample(df: pd.DataFrame, target_col: str, sample_size: int, random_state: int) -> pd.DataFrame:
        if len(df) <= sample_size:
            return df.copy()
        sampled_parts = []
        ratios = df[target_col].value_counts(normalize=True)
        for class_name, ratio in ratios.items():
            class_df = df[df[target_col] == class_name]
            n_samples = max(1, int(sample_size * ratio))
            sampled_parts.append(class_df.sample(n=min(n_samples, len(class_df)), random_state=random_state))
        return pd.concat(sampled_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

    @staticmethod
    def get_experiments() -> dict[str, dict[str, Any]]:
        return {
            "review_only": {"text_col": "text_review_only", "numeric_cols": []},
            "review_plus_categories": {"text_col": "text_review_categories", "numeric_cols": []},
            "review_plus_categories_tips": {"text_col": "text_review_categories_tips", "numeric_cols": []},
            "full_context_with_numeric": {
                "text_col": "text_full_context",
                "numeric_cols": ["review_word_count", "review_char_count", "tip_count", "avg_tip_compliment"],
            },
        }

    @staticmethod
    def build_pipeline(text_col: str, numeric_cols: list[str] | None = None) -> Pipeline:
        numeric_cols = numeric_cols or []
        transformers: list[tuple[str, Any, Any]] = [
            ("text", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, max_features=100_000), text_col)
        ]
        if numeric_cols:
            transformers.append(("numeric", StandardScaler(), numeric_cols))
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
        ])

    @staticmethod
    def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        preds = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "weighted_f1": f1_score(y_test, preds, average="weighted"),
            "classification_report": classification_report(y_test, preds, output_dict=True),
            "predictions": preds,
        }

    def run_ablation(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
        target_col = self.config.raw["model"]["target_col"]
        train_sample = self.stratified_sample(train_df, target_col, int(self.config.raw["data"]["train_sample_size"]), self.config.random_state)
        y_train, y_test = train_sample[target_col], test_df[target_col]
        results, trained_models = [], {}
        for experiment_name, experiment_config in self.get_experiments().items():
            text_col = experiment_config["text_col"]
            numeric_cols = experiment_config["numeric_cols"]
            feature_cols = [text_col] + numeric_cols
            model = self.build_pipeline(text_col=text_col, numeric_cols=numeric_cols)
            model.fit(train_sample[feature_cols], y_train)
            metrics = self.evaluate(model, test_df[feature_cols], y_test)
            results.append({
                "model_name": experiment_name,
                "text_col": text_col,
                "numeric_cols": ",".join(numeric_cols),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            })
            trained_models[experiment_name] = model
        return pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True), trained_models

    def tune_best_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame, best_experiment_name: str) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
        target_col = self.config.raw["model"]["target_col"]
        train_sample = self.stratified_sample(train_df, target_col, int(self.config.raw["data"]["train_sample_size"]), self.config.random_state)
        best_config = self.get_experiments()[best_experiment_name]
        text_col, numeric_cols = best_config["text_col"], best_config["numeric_cols"]
        feature_cols = [text_col] + numeric_cols
        pipeline = self.build_pipeline(text_col=text_col, numeric_cols=numeric_cols)
        raw_grid = self.config.raw["model"]["grid_search"]["param_grid"]
        param_grid = {
            "preprocessor__text__ngram_range": [tuple(x) for x in raw_grid["preprocessor__text__ngram_range"]],
            "preprocessor__text__min_df": raw_grid["preprocessor__text__min_df"],
            "preprocessor__text__max_features": raw_grid["preprocessor__text__max_features"],
            "classifier__C": raw_grid["classifier__C"],
        }
        grid_cfg = self.config.raw["model"]["grid_search"]
        grid_search = GridSearchCV(pipeline, param_grid, scoring=grid_cfg["scoring"], cv=int(grid_cfg["cv"]), n_jobs=int(grid_cfg["n_jobs"]), verbose=int(grid_cfg["verbose"]))
        grid_search.fit(train_sample[feature_cols], train_sample[target_col])
        best_model = grid_search.best_estimator_
        preds = best_model.predict(test_df[feature_cols])
        test_output = test_df.copy()
        test_output["predicted_sentiment"] = preds
        if hasattr(best_model.named_steps["classifier"], "predict_proba"):
            proba = best_model.predict_proba(test_df[feature_cols])
            for idx, class_name in enumerate(best_model.named_steps["classifier"].classes_):
                test_output[f"prob_{class_name}"] = proba[:, idx]
        metrics = self.evaluate(best_model, test_df[feature_cols], test_df[target_col])
        metrics.update({"best_params": grid_search.best_params_, "best_cv_score": grid_search.best_score_, "feature_cols": feature_cols})
        return best_model, test_output, metrics

    @staticmethod
    def feature_importance(model: Pipeline, top_n: int = 30) -> dict[str, pd.DataFrame]:
        vectorizer = model.named_steps["preprocessor"].named_transformers_["text"]
        classifier = model.named_steps["classifier"]
        feature_names = vectorizer.get_feature_names_out()
        tables = {}
        for class_idx, class_name in enumerate(classifier.classes_):
            coefs = classifier.coef_[class_idx]
            top_idx = np.argsort(coefs)[-top_n:][::-1]
            tables[class_name] = pd.DataFrame({"term": feature_names[top_idx], "coefficient": coefs[top_idx]})
        return tables

    @staticmethod
    def confusion_matrix_df(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"actual_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
        return cm_df, cm_df.div(cm_df.sum(axis=1), axis=0).round(4)
