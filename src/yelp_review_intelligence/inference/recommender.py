from __future__ import annotations

from dataclasses import dataclass

from yelp_review_intelligence.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class ActionRecommender:
    """
    Maps detected business topics to recommended operational actions.
    """

    action_map: dict[str, str] | None = None

    def __post_init__(self) -> None:
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
        try:
            recommendations = []

            for topic in topics:
                action = self.action_map.get(topic)

                if action and action not in recommendations:
                    recommendations.append(action)

            return recommendations

        except Exception as exc:
            logger.exception("Action recommendation failed.")
            raise RuntimeError("Action recommendation failed.") from exc