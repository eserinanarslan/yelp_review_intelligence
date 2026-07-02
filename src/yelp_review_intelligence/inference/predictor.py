from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.exceptions import ModelPersistenceError
from yelp_review_intelligence.logger import setup_logger
from yelp_review_intelligence.utils import load_model


# Configure project logger
logger = setup_logger(__name__)


@dataclass
class SentimentPredictor:
    """
    Prediction engine responsible for loading the trained model
    and performing sentiment inference.

    The model is loaded once during application startup and reused
    for all incoming API requests, reducing latency and avoiding
    repeated disk I/O.
    """

    config: ProjectConfig
    model: Any

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "SentimentPredictor":
        """
        Factory method for loading the production model.
        """
        model_path = config.model_dir / "sentiment_model.pkl"

        try:
            logger.info("Loading sentiment model from %s", model_path)

            # Load the serialized production model
            model = load_model(model_path)

            logger.info("Sentiment model loaded successfully.")

            return cls(config=config, model=model)

        except Exception as exc:
            logger.exception("Failed to load sentiment model.")

            raise ModelPersistenceError(
                f"Could not load model from {model_path}"
            ) from exc

    @staticmethod
    def build_input_frame(
        review_text: str,
        categories: str = "",
        sample_tips: str = "",
        city: str = "unknown",
        state: str = "unknown",
    ) -> pd.DataFrame:
        """
        Construct a single-row feature dataframe that matches
        the exact schema used during model training.

        Reusing the same feature engineering logic during
        inference guarantees consistency between training
        and production.
        """

        # Handle missing optional inputs
        review_text = review_text or ""
        categories = categories or ""
        sample_tips = sample_tips or ""
        city = city or "unknown"
        state = state or "unknown"

        # Generate numerical features
        review_word_count = len(review_text.split())
        review_char_count = len(review_text)
        tip_count = 1 if sample_tips.strip() else 0
        avg_tip_compliment = 0.0

        # Build different text representations
        text_review_only = review_text

        text_review_categories = (
            "Review: " + review_text +
            "\nCategories: " + categories
        )

        text_review_categories_tips = (
            text_review_categories +
            "\nTips: " + sample_tips
        )

        text_full_context = (
            text_review_categories_tips +
            "\nLocation: " + city + ", " + state
        )

        # Return the feature vector expected by the trained model
        return pd.DataFrame(
            [
                {
                    "text_review_only": text_review_only,
                    "text_review_categories": text_review_categories,
                    "text_review_categories_tips": text_review_categories_tips,
                    "text_full_context": text_full_context,
                    "review_word_count": review_word_count,
                    "review_char_count": review_char_count,
                    "tip_count": tip_count,
                    "avg_tip_compliment": avg_tip_compliment,
                }
            ]
        )

    def predict(
        self,
        review_text: str,
        categories: str = "",
        sample_tips: str = "",
        city: str = "unknown",
        state: str = "unknown",
    ) -> dict[str, Any]:
        """
        Predict the sentiment of a single customer review.

        Returns:
        - Predicted sentiment
        - Prediction confidence
        - Class probability distribution
        """

        try:
            # Build inference features
            input_df = self.build_input_frame(
                review_text=review_text,
                categories=categories,
                sample_tips=sample_tips,
                city=city,
                state=state,
            )

            # Run model inference
            prediction = self.model.predict(input_df)[0]

            confidence = None
            probabilities = {}

            # Compute prediction probabilities if supported
            if hasattr(self.model, "predict_proba"):

                proba = self.model.predict_proba(input_df)[0]

                classes = self.model.named_steps["classifier"].classes_

                probabilities = {
                    class_name: float(prob)
                    for class_name, prob in zip(classes, proba)
                }

                confidence = float(np.max(proba))

            # Return standardized prediction output
            return {
                "predicted_sentiment": str(prediction),
                "confidence": confidence,
                "probabilities": probabilities,
            }

        except Exception as exc:
            logger.exception("Sentiment prediction failed.")

            raise RuntimeError(
                "Sentiment prediction failed."
            ) from exc