from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from .config import ProjectConfig


@dataclass
class YelpDataPreparation:
    """
    End-to-end data preparation pipeline for the Yelp Review Intelligence project.

    Responsibilities:
    - Read raw Yelp JSON files
    - Create sentiment labels
    - Apply chronological train/test split
    - Aggregate historical tip information
    - Merge review, business and tip data
    - Generate modeling features
    """

    config: ProjectConfig

    def read_reviews(self) -> pd.DataFrame:
        """
        Read the most recent Yelp reviews from the raw review JSON file.

        DuckDB is used because it can efficiently query large JSON files
        without loading the entire file into memory at once.
        """
        review_path = self.config.raw_file_path("reviews")
        review_limit = int(self.config.raw["data"]["review_limit"])

        reviews = duckdb.sql(f"""
            SELECT
                review_id,
                user_id,
                business_id,
                stars AS review_stars,
                text AS review_text,
                date AS review_date
            FROM read_json_auto('{review_path}')
            ORDER BY date DESC
            LIMIT {review_limit}
        """).df()

        reviews["review_date"] = pd.to_datetime(reviews["review_date"])

        return reviews

    def read_business(self) -> pd.DataFrame:
        """
        Read business metadata.

        These fields provide contextual information such as business name,
        location and categories.
        """
        business_path = self.config.raw_file_path("business")

        return duckdb.sql(f"""
            SELECT
                business_id,
                name AS business_name,
                city,
                state,
                categories
            FROM read_json_auto('{business_path}')
        """).df()

    def read_tips(self) -> pd.DataFrame:
        """
        Read customer tip data.

        Tips are later aggregated at business level and used as historical
        contextual information.
        """
        tip_path = self.config.raw_file_path("tips")

        tips = duckdb.sql(f"""
            SELECT
                business_id,
                text AS tip_text,
                date AS tip_date,
                compliment_count AS tip_compliment_count
            FROM read_json_auto('{tip_path}')
        """).df()

        tips["tip_date"] = pd.to_datetime(tips["tip_date"])

        return tips

    @staticmethod
    def create_sentiment_target(reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Yelp star ratings into sentiment classes.

        1-2 stars -> negative
        3 stars   -> neutral
        4-5 stars -> positive
        """
        reviews = reviews.copy()

        reviews["target_sentiment"] = np.where(
            reviews["review_stars"] <= 2,
            "negative",
            np.where(
                reviews["review_stars"] == 3,
                "neutral",
                "positive",
            ),
        )

        return reviews

    def chronological_split(
        self,
        reviews: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        """
        Split reviews chronologically into train and test sets.

        This better simulates a real production scenario where future reviews
        are not available during training.
        """
        train_ratio = float(self.config.raw["data"]["train_ratio"])

        reviews = reviews.sort_values("review_date").reset_index(drop=True)

        split_idx = int(len(reviews) * train_ratio)

        train_reviews = reviews.iloc[:split_idx].copy()
        test_reviews = reviews.iloc[split_idx:].copy()

        # This timestamp is used as a cut-off point for leakage-safe tip aggregation.
        train_end_date = train_reviews["review_date"].max()

        return train_reviews, test_reviews, train_end_date

    @staticmethod
    def aggregate_tips(
        tips: pd.DataFrame,
        train_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Aggregate customer tips at business level.

        Leakage control:
        Only tips available up to the training cut-off date are used.
        This prevents future information from leaking into model training.
        """
        tips_train_context = tips[tips["tip_date"] <= train_end_date].copy()

        tips_agg = (
            tips_train_context
            .groupby("business_id")
            .agg(
                tip_count=("tip_text", "count"),
                sample_tips=(
                    "tip_text",
                    lambda x: " | ".join(
                        x.dropna().astype(str).head(3)
                    ),
                ),
                avg_tip_compliment=("tip_compliment_count", "mean"),
            )
            .reset_index()
        )

        return tips_agg

    @staticmethod
    def build_model_frame(
        review_df: pd.DataFrame,
        business_df: pd.DataFrame,
        tips_agg_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge reviews with business metadata and aggregated tip features.
        """
        return (
            review_df
            .merge(
                business_df,
                on="business_id",
                how="left",
            )
            .merge(
                tips_agg_df,
                on="business_id",
                how="left",
            )
        )

    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic numerical and cleaned text features.
        """
        df = df.copy()

        # Normalize missing text fields
        df["review_text"] = df["review_text"].fillna("").astype(str)
        df["categories"] = df["categories"].fillna("").astype(str)
        df["sample_tips"] = df["sample_tips"].fillna("").astype(str)
        df["city"] = df["city"].fillna("unknown").astype(str)
        df["state"] = df["state"].fillna("unknown").astype(str)

        # Basic review length features
        df["review_word_count"] = (
            df["review_text"]
            .str.count(r"\S+")
            .astype("int32")
        )

        df["review_char_count"] = (
            df["review_text"]
            .str.len()
            .astype("int32")
        )

        # Tip-related features
        df["tip_count"] = df["tip_count"].fillna(0).astype("int32")
        df["avg_tip_compliment"] = (
            df["avg_tip_compliment"]
            .fillna(0)
            .astype("float32")
        )

        return df

    @staticmethod
    def add_text_variants(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create different text representations for ablation experiments.

        These variants allow the modeling pipeline to compare whether
        categories, tips and location context improve prediction performance.
        """
        df = df.copy()

        df["text_review_only"] = df["review_text"]

        df["text_review_categories"] = (
            "Review: " + df["review_text"] +
            "\nCategories: " + df["categories"]
        )

        df["text_review_categories_tips"] = (
            df["text_review_categories"] +
            "\nTips: " + df["sample_tips"]
        )

        df["text_full_context"] = (
            df["text_review_categories_tips"] +
            "\nLocation: " + df["city"] + ", " + df["state"]
        )

        return df

    @staticmethod
    def select_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only the columns required for downstream model training.
        """
        selected_columns = [
            "review_id",
            "business_id",
            "user_id",
            "review_date",
            "review_stars",
            "target_sentiment",
            "review_text",
            "text_review_only",
            "text_review_categories",
            "text_review_categories_tips",
            "text_full_context",
            "business_name",
            "city",
            "state",
            "categories",
            "sample_tips",
            "review_word_count",
            "review_char_count",
            "tip_count",
            "avg_tip_compliment",
        ]

        return df[selected_columns].copy()

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete data preparation pipeline.
        """
        # Read and label review data
        reviews = self.create_sentiment_target(
            self.read_reviews()
        )

        # Read contextual datasets
        business = self.read_business()
        tips = self.read_tips()

        # Create leakage-safe chronological split
        train_reviews, test_reviews, train_end_date = self.chronological_split(
            reviews
        )

        # Aggregate historical tips using only information available before train cut-off
        tips_agg = self.aggregate_tips(
            tips,
            train_end_date,
        )

        # Merge datasets into train and test modeling frames
        train_df = self.build_model_frame(
            train_reviews,
            business,
            tips_agg,
        )

        test_df = self.build_model_frame(
            test_reviews,
            business,
            tips_agg,
        )

        # Add engineered features and text variants
        train_df = self.add_text_variants(
            self.add_basic_features(train_df)
        )

        test_df = self.add_text_variants(
            self.add_basic_features(test_df)
        )

        # Return final train and test datasets
        return (
            self.select_modeling_columns(train_df),
            self.select_modeling_columns(test_df),
        )