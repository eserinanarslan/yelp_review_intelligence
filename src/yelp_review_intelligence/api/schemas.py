from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length=1,
        description="Customer review text.",
    )
    categories: Optional[str] = Field(
        default="",
        description="Business categories, such as Restaurants, Italian, Pizza.",
    )
    sample_tips: Optional[str] = Field(
        default="",
        description="Optional existing customer tips for the business.",
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
    predicted_sentiment: str
    confidence: Optional[float]
    probabilities: dict[str, float]
    topics: list[str]
    severity: str
    recommended_actions: list[str]


class BatchReviewRequest(BaseModel):
    reviews: list[ReviewRequest]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    description: str