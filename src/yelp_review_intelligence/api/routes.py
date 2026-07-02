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


# Configure project logger
logger = setup_logger(__name__)

# Create API router
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Used by monitoring tools (Docker, Kubernetes, Load Balancers)
    to verify that the service is running.
    """
    return HealthResponse(status="healthy")


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """
    Return metadata about the deployed production model.
    Useful for debugging, monitoring and version tracking.
    """
    return ModelInfoResponse(
        model_name="Yelp Review Sentiment Model",
        version="1.0.0",
        description="TF-IDF + Logistic Regression model for customer review sentiment analysis.",
    )


@router.post("/analyze-review", response_model=ReviewAnalysisResponse)
def analyze_review(
    request: Request,
    payload: ReviewRequest,
) -> ReviewAnalysisResponse:
    """
    Analyze a single customer review.

    The API layer only validates the request and delegates
    all business logic to the ReviewAnalysisService.
    """
    try:
        # Retrieve the shared service initialized during application startup
        service = request.app.state.review_service

        # Execute the complete review analysis pipeline
        result = service.analyze_review(
            review_text=payload.review_text,
            categories=payload.categories or "",
            sample_tips=payload.sample_tips or "",
            city=payload.city or "unknown",
            state=payload.state or "unknown",
        )

        # Return a structured API response
        return ReviewAnalysisResponse(**result)

    except Exception as exc:
        # Log unexpected API errors
        logger.exception("API request failed: /analyze-review")

        raise HTTPException(
            status_code=500,
            detail="Review analysis failed.",
        ) from exc


@router.post("/analyze-batch", response_model=list[ReviewAnalysisResponse])
def analyze_batch(
    request: Request,
    payload: BatchReviewRequest,
) -> list[ReviewAnalysisResponse]:
    """
    Analyze multiple customer reviews in a single request.

    Batch inference reduces network overhead and improves
    throughput when processing large numbers of reviews.
    """
    try:
        # Retrieve the shared analysis service
        service = request.app.state.review_service

        # Convert validated request objects into dictionaries
        reviews = [
            review.model_dump()
            for review in payload.reviews
        ]

        # Run batch prediction
        results = service.analyze_batch(reviews)

        # Convert results into API response models
        return [
            ReviewAnalysisResponse(**result)
            for result in results
        ]

    except Exception as exc:
        # Log unexpected API errors
        logger.exception("API request failed: /analyze-batch")

        raise HTTPException(
            status_code=500,
            detail="Batch review analysis failed.",
        ) from exc