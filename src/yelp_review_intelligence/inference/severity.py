from __future__ import annotations

from dataclasses import dataclass

from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.logger import setup_logger


# Configure project logger
logger = setup_logger(__name__)


@dataclass
class SeverityScorer:
    """
    Business rule engine that estimates how critical a review is.

    Severity is determined using the predicted sentiment together
    with additional heuristics such as review length and the
    presence of high-severity keywords.
    """

    config: ProjectConfig

    def score(
        self,
        review_text: str,
        predicted_sentiment: str,
        review_word_count: int | None = None,
    ) -> str:
        """
        Calculate the severity level of a customer review.

        Returns:
        - low
        - medium
        - high
        """

        try:
            # Normalize review text
            text = str(review_text or "").lower()

            # Positive reviews are always considered low severity
            if predicted_sentiment == "positive":
                return "low"

            severity_score = 0

            # Negative reviews start with a higher base score
            if predicted_sentiment == "negative":
                severity_score += 2

            # Longer reviews often indicate more detailed complaints
            threshold = int(
                self.config.raw["severity"]["long_review_threshold"]
            )

            if (
                review_word_count is not None
                and review_word_count > threshold
            ):
                severity_score += 1

            # Increase severity when high-risk keywords are detected
            for pattern in self.config.raw["severity"]["high_severity_patterns"]:

                if str(pattern).lower() in text:
                    severity_score += 1

            # Convert the numeric score into a business-friendly label
            if severity_score >= 3:
                return "high"

            if severity_score == 2:
                return "medium"

            return "low"

        except Exception as exc:
            logger.exception("Severity scoring failed.")

            raise RuntimeError(
                "Severity scoring failed."
            ) from exc