from __future__ import annotations
from dataclasses import dataclass
import duckdb
import numpy as np
import pandas as pd
from .config import ProjectConfig


@dataclass
class YelpDataPreparation:
    config: ProjectConfig

    def read_reviews(self) -> pd.DataFrame:
        review_path = self.config.raw_file_path("reviews")
        review_limit = int(self.config.raw["data"]["review_limit"])
        reviews = duckdb.sql(f'''
            SELECT review_id, user_id, business_id, stars AS review_stars,
                   text AS review_text, date AS review_date
            FROM read_json_auto('{review_path}')
            ORDER BY date DESC
            LIMIT {review_limit}
        ''').df()
        reviews["review_date"] = pd.to_datetime(reviews["review_date"])
        return reviews

    def read_business(self) -> pd.DataFrame:
        business_path = self.config.raw_file_path("business")
        return duckdb.sql(f'''
            SELECT business_id, name AS business_name, city, state, categories
            FROM read_json_auto('{business_path}')
        ''').df()

    def read_tips(self) -> pd.DataFrame:
        tip_path = self.config.raw_file_path("tips")
        tips = duckdb.sql(f'''
            SELECT business_id, text AS tip_text, date AS tip_date,
                   compliment_count AS tip_compliment_count
            FROM read_json_auto('{tip_path}')
        ''').df()
        tips["tip_date"] = pd.to_datetime(tips["tip_date"])
        return tips

    @staticmethod
    def create_sentiment_target(reviews: pd.DataFrame) -> pd.DataFrame:
        reviews = reviews.copy()
        reviews["target_sentiment"] = np.where(
            reviews["review_stars"] <= 2,
            "negative",
            np.where(reviews["review_stars"] == 3, "neutral", "positive"),
        )
        return reviews

    def chronological_split(self, reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        train_ratio = float(self.config.raw["data"]["train_ratio"])
        reviews = reviews.sort_values("review_date").reset_index(drop=True)
        split_idx = int(len(reviews) * train_ratio)
        train_reviews = reviews.iloc[:split_idx].copy()
        test_reviews = reviews.iloc[split_idx:].copy()
        return train_reviews, test_reviews, train_reviews["review_date"].max()

    @staticmethod
    def aggregate_tips(tips: pd.DataFrame, train_end_date: pd.Timestamp) -> pd.DataFrame:
        # Leakage control: only use tips available up to train cut-off date.
        tips_train_context = tips[tips["tip_date"] <= train_end_date].copy()
        return (
            tips_train_context.groupby("business_id")
            .agg(
                tip_count=("tip_text", "count"),
                sample_tips=("tip_text", lambda x: " | ".join(x.dropna().astype(str).head(3))),
                avg_tip_compliment=("tip_compliment_count", "mean"),
            )
            .reset_index()
        )

    @staticmethod
    def build_model_frame(review_df: pd.DataFrame, business_df: pd.DataFrame, tips_agg_df: pd.DataFrame) -> pd.DataFrame:
        return review_df.merge(business_df, on="business_id", how="left").merge(tips_agg_df, on="business_id", how="left")

    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["review_text"] = df["review_text"].fillna("").astype(str)
        df["categories"] = df["categories"].fillna("").astype(str)
        df["sample_tips"] = df["sample_tips"].fillna("").astype(str)
        df["city"] = df["city"].fillna("unknown").astype(str)
        df["state"] = df["state"].fillna("unknown").astype(str)
        df["review_word_count"] = df["review_text"].str.count(r"\S+").astype("int32")
        df["review_char_count"] = df["review_text"].str.len().astype("int32")
        df["tip_count"] = df["tip_count"].fillna(0).astype("int32")
        df["avg_tip_compliment"] = df["avg_tip_compliment"].fillna(0).astype("float32")
        return df

    @staticmethod
    def add_text_variants(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["text_review_only"] = df["review_text"]
        df["text_review_categories"] = "Review: " + df["review_text"] + "\nCategories: " + df["categories"]
        df["text_review_categories_tips"] = df["text_review_categories"] + "\nTips: " + df["sample_tips"]
        df["text_full_context"] = df["text_review_categories_tips"] + "\nLocation: " + df["city"] + ", " + df["state"]
        return df

    @staticmethod
    def select_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
        selected_columns = [
            "review_id", "business_id", "user_id", "review_date", "review_stars",
            "target_sentiment", "review_text", "text_review_only", "text_review_categories",
            "text_review_categories_tips", "text_full_context", "business_name", "city",
            "state", "categories", "sample_tips", "review_word_count", "review_char_count",
            "tip_count", "avg_tip_compliment",
        ]
        return df[selected_columns].copy()

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        reviews = self.create_sentiment_target(self.read_reviews())
        business = self.read_business()
        tips = self.read_tips()
        train_reviews, test_reviews, train_end_date = self.chronological_split(reviews)
        tips_agg = self.aggregate_tips(tips, train_end_date)
        train_df = self.build_model_frame(train_reviews, business, tips_agg)
        test_df = self.build_model_frame(test_reviews, business, tips_agg)
        train_df = self.add_text_variants(self.add_basic_features(train_df))
        test_df = self.add_text_variants(self.add_basic_features(test_df))
        return self.select_modeling_columns(train_df), self.select_modeling_columns(test_df)
