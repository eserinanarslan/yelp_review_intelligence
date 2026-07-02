from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from .config import ProjectConfig


@dataclass
class ReviewIntelligenceGenerator:
    """
    Generates business-facing review intelligence outputs from model predictions.

    This class is used after model inference to convert raw predictions into
    dashboard-ready insights such as topics, severity levels, representative
    reviews, and recommended actions.
    """

    config: ProjectConfig

    def detect_topics(self, text: str) -> list[str]:
        """
        Detect business-relevant topics from review text using keyword rules.

        This is intentionally simple and interpretable for the first version.
        It can later be replaced by BERTopic, zero-shot classification,
        or an LLM-based topic detector.
        """
        text = str(text).lower()
        detected_topics = []

        # Iterate over predefined topic-keyword mappings from the config file
        for topic, keywords in self.config.raw["topics"].items():
            for keyword in keywords:
                pattern = r"\b" + re.escape(str(keyword).lower()) + r"\b"

                if re.search(pattern, text):
                    detected_topics.append(topic)
                    break

        # Assign a fallback topic when no specific topic is found
        return detected_topics if detected_topics else ["Other"]

    def calculate_severity(self, row: pd.Series) -> str:
        """
        Estimate the operational severity of a review.

        Severity is based on predicted sentiment, review length,
        and high-severity complaint patterns.
        """
        text = str(row["review_text"]).lower()
        sentiment = row["predicted_sentiment"]

        # Positive reviews are considered low severity
        if sentiment == "positive":
            return "low"

        # Negative reviews start with a higher base severity score
        score = 2 if sentiment == "negative" else 0

        # Longer reviews may indicate more detailed complaints
        if int(row.get("review_word_count", 0)) > int(
            self.config.raw["severity"]["long_review_threshold"]
        ):
            score += 1

        # Add severity points when critical patterns are mentioned
        for pattern in self.config.raw["severity"]["high_severity_patterns"]:
            if str(pattern).lower() in text:
                score += 1

        # Convert numeric score into business-friendly labels
        if score >= 3:
            return "high"

        if score == 2:
            return "medium"

        return "low"

    def create_topic_level_df(
        self,
        predictions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Expand prediction results into topic-level rows.

        One review may mention multiple topics. Exploding the topic list
        makes it possible to aggregate complaints by topic later.
        """
        df = predictions_df.copy()

        # Detect topics for each review
        df["detected_topics"] = df["review_text"].apply(self.detect_topics)

        # Convert one review with multiple topics into multiple rows
        topic_level_df = (
            df.explode("detected_topics")
            .rename(columns={"detected_topics": "topic"})
            .reset_index(drop=True)
        )

        # Add severity label for each topic-level record
        topic_level_df["severity"] = topic_level_df.apply(
            self.calculate_severity,
            axis=1,
        )

        return topic_level_df

    @staticmethod
    def create_business_insights(
        topic_level_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate topic-level predictions into business-level insights.

        The output can be used directly in dashboards or business reports.
        """
        insights = (
            topic_level_df
            .groupby(
                [
                    "business_id",
                    "business_name",
                    "city",
                    "state",
                    "topic",
                ]
            )
            .agg(
                review_count=("review_id", "count"),
                negative_count=(
                    "predicted_sentiment",
                    lambda x: (x == "negative").sum(),
                ),
                neutral_count=(
                    "predicted_sentiment",
                    lambda x: (x == "neutral").sum(),
                ),
                positive_count=(
                    "predicted_sentiment",
                    lambda x: (x == "positive").sum(),
                ),
                high_severity_count=(
                    "severity",
                    lambda x: (x == "high").sum(),
                ),
            )
            .reset_index()
        )

        # Negative ratio helps prioritize problematic topics
        insights["negative_ratio"] = (
            insights["negative_count"] / insights["review_count"]
        )

        return insights

    @staticmethod
    def action_map() -> dict[str, str]:
        """
        Define business recommendations for each detected topic.
        """
        return {
            "Waiting Time": "Investigate staffing and table turnover during peak hours.",
            "Food Quality": "Review kitchen quality control and food temperature before serving.",
            "Service": "Review staff training and manager escalation process.",
            "Price": "Check whether customers perceive the offer as good value for money.",
            "Cleanliness": "Audit cleaning routines, especially restrooms and high-traffic areas.",
            "Atmosphere": "Review noise level, music, seating comfort, and ambience.",
            "Parking": "Improve parking instructions or communicate nearby parking alternatives.",
            "Other": "Manually review representative negative reviews to identify emerging issues.",
        }

    @staticmethod
    def pick_example_business(
        predictions_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Select an example business with the highest number of analyzed reviews.

        This is useful for generating a representative dashboard example
        when no specific business_id is provided.
        """
        return (
            predictions_df
            .groupby(
                [
                    "business_id",
                    "business_name",
                    "city",
                    "state",
                ]
            )
            .agg(review_count=("review_id", "count"))
            .reset_index()
            .sort_values("review_count", ascending=False)
            .iloc[0]
        )

    def create_dashboard_output(
        self,
        predictions_df: pd.DataFrame,
        topic_level_df: pd.DataFrame,
        business_id: str | None = None,
    ):
        """
        Create dashboard-ready outputs for a selected business.

        Returns:
        - JSON-style dashboard output
        - Overall sentiment distribution
        - Topic summary table
        - Representative negative reviews
        """

        # Select an example business if no business_id is provided
        if business_id is None:
            example_business = self.pick_example_business(predictions_df)
            business_id = str(example_business["business_id"])
        else:
            example_business = (
                predictions_df[predictions_df["business_id"] == business_id]
                .iloc[0]
            )

        # Filter reviews and topic-level records for the selected business
        example_reviews = predictions_df[
            predictions_df["business_id"] == business_id
        ].copy()

        example_topic_rows = topic_level_df[
            topic_level_df["business_id"] == business_id
        ].copy()

        # Calculate sentiment distribution
        overall_sentiment = (
            example_reviews["predicted_sentiment"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .reset_index()
        )

        overall_sentiment.columns = ["sentiment", "percentage"]

        # Summarize topic-level business issues
        topic_summary = (
            example_topic_rows
            .groupby("topic")
            .agg(
                review_count=("review_id", "count"),
                negative_ratio=(
                    "predicted_sentiment",
                    lambda x: (x == "negative").mean(),
                ),
                high_severity_count=(
                    "severity",
                    lambda x: (x == "high").sum(),
                ),
            )
            .reset_index()
            .sort_values(
                [
                    "high_severity_count",
                    "negative_ratio",
                    "review_count",
                ],
                ascending=False,
            )
        )

        # Add recommended business action for each topic
        topic_summary["recommended_action"] = (
            topic_summary["topic"].map(self.action_map())
        )

        # Select top topics for representative review extraction
        top_topics = topic_summary.head(3)["topic"].tolist()

        # Pick representative negative reviews for the most important topics
        representative_reviews = (
            example_topic_rows[
                (example_topic_rows["topic"].isin(top_topics))
                & (example_topic_rows["predicted_sentiment"] == "negative")
            ]
            .sort_values(
                [
                    "severity",
                    "review_word_count",
                ],
                ascending=False,
            )
            [
                [
                    "topic",
                    "severity",
                    "review_text",
                    "predicted_sentiment",
                    "review_date",
                ]
            ]
            .head(10)
        )

        # Build final dashboard-ready JSON output
        dashboard_output = {
            "business_name": example_business["business_name"],
            "city": example_business["city"],
            "state": example_business["state"],
            "number_of_reviews_analyzed": len(example_reviews),
            "overall_sentiment": overall_sentiment.to_dict(
                orient="records"
            ),
            "top_topics": topic_summary.head(10).to_dict(
                orient="records"
            ),
            "representative_negative_reviews": (
                representative_reviews.to_dict(orient="records")
            ),
        }

        return (
            dashboard_output,
            overall_sentiment,
            topic_summary,
            representative_reviews,
        )