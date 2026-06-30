from __future__ import annotations

import re
from dataclasses import dataclass

from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class TopicDetector:
    """
    Detects business-relevant topics from review text.

    This first version uses interpretable keyword rules.
    It can later be replaced with BERTopic, zero-shot classification,
    or a supervised topic model.
    """

    config: ProjectConfig

    def detect(self, text: str) -> list[str]:
        try:
            text = str(text or "").lower()
            detected_topics = []

            for topic, keywords in self.config.raw["topics"].items():
                for keyword in keywords:
                    pattern = r"\b" + re.escape(str(keyword).lower()) + r"\b"

                    if re.search(pattern, text):
                        detected_topics.append(topic)
                        break

            return detected_topics if detected_topics else ["Other"]

        except Exception as exc:
            logger.exception("Topic detection failed.")
            raise RuntimeError("Topic detection failed.") from exc