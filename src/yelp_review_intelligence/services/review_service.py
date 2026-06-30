from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yelp_review_intelligence.inference.predictor import SentimentPredictor
from yelp_review_intelligence.inference.recommender import ActionRecommender
from yelp_review_intelligence.inference.severity import SeverityScorer
from yelp_review_intelligence.inference.topic_detector import TopicDetector
from yelp_review_intelligence.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class ReviewAnalysisService:
    """
    Orchestrates the complete review intelligence pipeline.

    API layer calls this service instead of directly calling model
    or inference components.
    """

    predictor: SentimentPredictor
    topic_detector: TopicDetector
    severity_scorer: SeverityScorer
    recommender: ActionRecommender

    def analyze_review(
        self,
        review_text: str,
        categories: str = "",
        sample_tips: str = "",
        city: str = "unknown",
        state: str = "unknown",
    ) -> dict[str, Any]:
        try:
            logger.info("Analyzing single review.")

            sentiment_result = self.predictor.predict(
                review_text=review_text,
                categories=categories,
                sample_tips=sample_tips,
                city=city,
                state=state,
            )

            topics = self.topic_detector.detect(review_text)
            review_word_count = len(str(review_text or "").split())

            severity = self.severity_scorer.score(
                review_text=review_text,
                predicted_sentiment=sentiment_result["predicted_sentiment"],
                review_word_count=review_word_count,
            )

            recommended_actions = self.recommender.recommend(topics)

            return {
                "predicted_sentiment": sentiment_result["predicted_sentiment"],
                "confidence": sentiment_result["confidence"],
                "probabilities": sentiment_result["probabilities"],
                "topics": topics,
                "severity": severity,
                "recommended_actions": recommended_actions,
            }

        except Exception as exc:
            logger.exception("Review analysis failed.")
            raise RuntimeError("Review analysis failed.") from exc

    def analyze_batch(self, reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            logger.info("Analyzing batch of %s reviews.", len(reviews))

            results = []

            for review in reviews:
                result = self.analyze_review(
                    review_text=review.get("review_text", ""),
                    categories=review.get("categories", ""),
                    sample_tips=review.get("sample_tips", ""),
                    city=review.get("city", "unknown"),
                    state=review.get("state", "unknown"),
                )

                results.append(result)

            return results

        except Exception as exc:
            logger.exception("Batch review analysis failed.")
            raise RuntimeError("Batch review analysis failed.") from exc