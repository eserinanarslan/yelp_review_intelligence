from __future__ import annotations

from fastapi import Depends, Request

from yelp_review_intelligence.services.review_service import (
    ReviewAnalysisService,
)


def get_review_service(
    request: Request,
) -> ReviewAnalysisService:
    """
    Return the singleton review analysis service.

    The service is initialized once during FastAPI startup
    and reused for every request.
    """
    return request.app.state.review_service