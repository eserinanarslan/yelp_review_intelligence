from __future__ import annotations

import re
from dataclasses import dataclass

from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.logger import setup_logger


# Configure project logger
logger = setup_logger(__name__)


@dataclass
class TopicDetector:
    """
    Detect business-relevant complaint topics from review text.

    The current implementation uses simple keyword matching
    for transparency and fast inference. The component is
    intentionally isolated so it can later be replaced by
    more advanced NLP techniques such as BERTopic, zero-shot
    classification, or transformer-based models.
    """

    config: ProjectConfig

    def detect(self, text: str) -> list[str]:
        """
        Detect complaint topics mentioned in a customer review.

        Returns a list of detected business topics.
        """

        try:
            # Normalize the review text
            text = str(text or "").lower()

            detected_topics = []

            # Iterate through all predefined topics
            for topic, keywords in self.config.raw["topics"].items():

                # Search for any keyword associated with the topic
                for keyword in keywords:

                    pattern = r"\b" + re.escape(str(keyword).lower()) + r"\b"

                    if re.search(pattern, text):
                        detected_topics.append(topic)
                        break

            # Return a default category if no topic is detected
            return detected_topics if detected_topics else ["Other"]

        except Exception as exc:
            logger.exception("Topic detection failed.")

            raise RuntimeError(
                "Topic detection failed."
            ) from exc