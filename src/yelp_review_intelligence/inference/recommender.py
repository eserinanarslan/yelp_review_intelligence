from __future__ import annotations

from dataclasses import dataclass

from yelp_review_intelligence.logger import setup_logger


# Configure project logger
logger = setup_logger(__name__)


@dataclass
class ActionRecommender:
    """
    Business rule engine that converts detected complaint topics
    into actionable operational recommendations.

    This component is intentionally separated from the ML model,
    allowing business rules to evolve independently of model training.
    """

    action_map: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """
        Initialize the default topic-to-action mapping.
        """

        if self.action_map is None:
            self.action_map = {
                "Waiting Time": "Investigate staffing and table turnover during peak hours.",
                "Food Quality": "Review kitchen quality control and food temperature before serving.",
                "Service": "Review staff training and manager escalation process.",
                "Price": "Check whether customers perceive the offer as good value for money.",
                "Cleanliness": "Audit cleaning routines, especially restrooms and high-traffic areas.",
                "Atmosphere": "Review noise level, music, seating comfort, and ambience.",
                "Parking": "Improve parking instructions or communicate nearby parking alternatives.",
                "Other": "Manually review the review text to identify emerging issues.",
            }

    def recommend(self, topics: list[str]) -> list[str]:
        """
        Generate business recommendations based on
        the detected complaint topics.
        """

        try:
            recommendations = []

            # Convert each detected topic into a business action
            for topic in topics:

                action = self.action_map.get(topic)

                # Avoid duplicate recommendations
                if action and action not in recommendations:
                    recommendations.append(action)

            return recommendations

        except Exception as exc:
            logger.exception("Action recommendation failed.")

            raise RuntimeError(
                "Action recommendation failed."
            ) from exc