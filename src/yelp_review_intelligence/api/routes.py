from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from yelp_review_intelligence.api.schemas import (
    BatchReviewRequest,
    HealthResponse,
    ModelInfoResponse,
    ReviewAnalysisResponse,
    ReviewRequest,
)
from yelp_review_intelligence.logger import setup_logger


logger = setup_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health endpoint used by orchestration tools and monitoring systems.
    """
    return HealthResponse(status="healthy")


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """
    Return high-level model information.
    """
    return ModelInfoResponse(
        model_name="Yelp Review Sentiment Model",
        version="1.0.0",
        description="TF-IDF + Logistic Regression model for customer review sentiment analysis.",
    )


@router.post("/analyze-review", response_model=ReviewAnalysisResponse)
def analyze_review(request: Request, payload: ReviewRequest) -> ReviewAnalysisResponse:
    """
    Analyze a single customer review.
    """
    try:
        service = request.app.state.review_service

        result = service.analyze_review(
            review_text=payload.review_text,
            categories=payload.categories or "",
            sample_tips=payload.sample_tips or "",
            city=payload.city or "unknown",
            state=payload.state or "unknown",
        )

        return ReviewAnalysisResponse(**result)

    except Exception as exc:
        logger.exception("API request failed: /analyze-review")
        raise HTTPException(
            status_code=500,
            detail="Review analysis failed.",
        ) from exc


@router.post("/analyze-batch", response_model=list[ReviewAnalysisResponse])
def analyze_batch(request: Request, payload: BatchReviewRequest) -> list[ReviewAnalysisResponse]:
    """
    Analyze multiple reviews in a single request.
    """
    try:
        service = request.app.state.review_service

        reviews = [
            review.model_dump()
            for review in payload.reviews
        ]

        results = service.analyze_batch(reviews)

        return [
            ReviewAnalysisResponse(**result)
            for result in results
        ]

    except Exception as exc:
        logger.exception("API request failed: /analyze-batch")
        raise HTTPException(
            status_code=500,
            detail="Batch review analysis failed.",
        ) from exc