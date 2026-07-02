from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    """
    Input schema for analyzing a single customer review.

    Pydantic automatically validates incoming requests before
    they reach the business logic.
    """

    review_text: str = Field(
        ...,
        min_length=1,
        description="Customer review text.",
    )

    categories: Optional[str] = Field(
        default="",
        description="Business categories (e.g. Restaurants, Italian, Pizza).",
    )

    sample_tips: Optional[str] = Field(
        default="",
        description="Historical customer tips.",
    )

    city: Optional[str] = Field(
        default="unknown",
        description="Business city.",
    )

    state: Optional[str] = Field(
        default="unknown",
        description="Business state.",
    )


class ReviewAnalysisResponse(BaseModel):
    """
    Standard API response returned after review analysis.
    """

    predicted_sentiment: str
    confidence: Optional[float]
    probabilities: dict[str, float]
    topics: list[str]
    severity: str
    recommended_actions: list[str]


class BatchReviewRequest(BaseModel):
    """
    Input schema for batch inference.
    Allows multiple reviews to be analyzed in a single API call.
    """

    reviews: list[ReviewRequest]


class HealthResponse(BaseModel):
    """
    Response returned by the health check endpoint.
    """

    status: str


class ModelInfoResponse(BaseModel):
    """
    Response containing metadata about the deployed production model.
    """

    model_name: str
    version: str
    description: str