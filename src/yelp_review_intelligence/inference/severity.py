from __future__ import annotations

from dataclasses import dataclass

from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class SeverityScorer:
    """
    Scores review severity based on predicted sentiment and complaint signals.
    """

    config: ProjectConfig

    def score(
        self,
        review_text: str,
        predicted_sentiment: str,
        review_word_count: int | None = None,
    ) -> str:
        try:
            text = str(review_text or "").lower()

            if predicted_sentiment == "positive":
                return "low"

            severity_score = 0

            if predicted_sentiment == "negative":
                severity_score += 2

            threshold = int(self.config.raw["severity"]["long_review_threshold"])

            if review_word_count is not None and review_word_count > threshold:
                severity_score += 1

            for pattern in self.config.raw["severity"]["high_severity_patterns"]:
                if str(pattern).lower() in text:
                    severity_score += 1

            if severity_score >= 3:
                return "high"

            if severity_score == 2:
                return "medium"

            return "low"

        except Exception as exc:
            logger.exception("Severity scoring failed.")
            raise RuntimeError("Severity scoring failed.") from exc