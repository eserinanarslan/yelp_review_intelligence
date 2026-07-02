from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from yelp_review_intelligence.api.routes import router
from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.inference.predictor import SentimentPredictor
from yelp_review_intelligence.inference.recommender import ActionRecommender
from yelp_review_intelligence.inference.severity import SeverityScorer
from yelp_review_intelligence.inference.topic_detector import TopicDetector
from yelp_review_intelligence.logger import setup_logger
from yelp_review_intelligence.services.review_service import ReviewAnalysisService


# Configure project logger
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifecycle.

    Executed once when the application starts and once when it shuts down.

    Responsibilities:
    - Load configuration
    - Load the trained ML model
    - Initialize all business services
    - Register shared dependencies
    """
    try:
        logger.info("Starting Yelp Review Intelligence API.")

        # Load project configuration
        config = ProjectConfig.from_yaml("configs/config.yaml")

        # Initialize inference components
        predictor = SentimentPredictor.from_config(config)
        topic_detector = TopicDetector(config)
        severity_scorer = SeverityScorer(config)
        recommender = ActionRecommender()

        # Compose the complete review analysis service
        app.state.review_service = ReviewAnalysisService(
            predictor=predictor,
            topic_detector=topic_detector,
            severity_scorer=severity_scorer,
            recommender=recommender,
        )

        logger.info("API dependencies loaded successfully.")

        # Keep the application running
        yield

    except Exception:
        # Log startup failures
        logger.exception("API startup failed.")
        raise

    finally:
        # Executed during graceful shutdown
        logger.info("Shutting down Yelp Review Intelligence API.")


# Create the FastAPI application
app = FastAPI(
    title="Yelp Review Intelligence API",
    description="REST API for transforming customer reviews into actionable business insights.",
    version="1.0.0",
    lifespan=lifespan,
)

# Register all API endpoints
app.include_router(router)