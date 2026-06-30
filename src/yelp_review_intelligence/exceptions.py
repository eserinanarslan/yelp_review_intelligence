from __future__ import annotations


class YelpReviewIntelligenceError(Exception):
    """Base exception for the Yelp Review Intelligence project."""


class ConfigError(YelpReviewIntelligenceError):
    """Raised when configuration loading or validation fails."""


class DataPreparationError(YelpReviewIntelligenceError):
    """Raised when data preparation fails."""


class ModelTrainingError(YelpReviewIntelligenceError):
    """Raised when model training or evaluation fails."""


class IntelligenceGenerationError(YelpReviewIntelligenceError):
    """Raised when business insight generation fails."""


class ModelPersistenceError(YelpReviewIntelligenceError):
    """Raised when saving or loading model artifacts fails."""