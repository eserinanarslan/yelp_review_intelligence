# Yelp Review Intelligence

An end-to-end AI system that transforms unstructured customer reviews into actionable business intelligence for restaurant owners and multi-location restaurant operations teams.

This project was developed as a Data Science case study using the Yelp Academic Dataset. Instead of building only a sentiment classifier, the objective is to design a production-ready NLP solution capable of helping business stakeholders understand thousands of customer reviews without reading them individually.

---

# Business Problem

Restaurant managers receive hundreds or even thousands of customer reviews every month.

Reading every review is impossible, making it difficult to identify recurring operational issues and customer satisfaction trends.

This project transforms raw customer reviews into structured business insights by automatically identifying:

- Overall sentiment
- Complaint topics
- Review severity
- Representative customer reviews
- Recommended operational actions

The final outputs are designed to be consumed by dashboards, BI tools, REST APIs, or future AI assistants.

---

# Solution Overview

The complete pipeline consists of the following stages:

```
                   Yelp Dataset
                        │
                        ▼
              Data Preparation Pipeline
                        │
                        ▼
              Feature Engineering
                        │
                        ▼
             Sentiment Classification
                        │
                        ▼
              Topic Detection
                        │
                        ▼
              Severity Scoring
                        │
                        ▼
          Business Intelligence Layer
                        │
                        ▼
             FastAPI REST Service
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
     Dashboard                  Future Chatbot
```

---

# Features

## Data Preparation

- Reads Yelp Academic Dataset
- Handles large JSON files efficiently
- Leakage-safe chronological train/test split
- Business-level feature aggregation
- Tip aggregation before prediction date
- Feature engineering
- Text preprocessing

---

## Feature Engineering

Multiple feature configurations were evaluated through an ablation study.

The following feature sets were compared:

- Review text only
- Review + business categories
- Review + categories + tips
- Full contextual features

Additional engineered features include:

- Review length
- Character count
- Tip count
- Average tip compliments
- Business location

---

## Model Training

The training pipeline includes:

- TF-IDF Vectorization
- ColumnTransformer
- Logistic Regression
- GridSearchCV
- Cross Validation
- Feature Selection
- Hyperparameter Optimization
- Model Evaluation

Evaluation metrics include:

- Accuracy
- Macro F1
- Weighted F1
- Confusion Matrix
- Classification Report
- Feature Importance

---

## Business Intelligence

Instead of returning only sentiment predictions, the system generates business-oriented outputs.

For every review it produces:

- Predicted sentiment
- Prediction confidence
- Complaint topics
- Severity level
- Recommended actions

Business-level outputs include:

- Overall sentiment distribution
- Top complaint topics
- Representative reviews
- Dashboard-ready JSON
- CSV reports

---

# REST API

The trained model is exposed through a FastAPI service.

Available endpoints:

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | /health | Health check |
| GET | /model-info | Model metadata |
| POST | /analyze-review | Analyze a single review |
| POST | /analyze-batch | Analyze multiple reviews |

Swagger documentation:

```
http://localhost:8000/docs
```

---

# Project Structure

```
.
├── configs/
│   └── config.yaml
│
├── data/
│   └── processed/
│
├── models/
│   └── sentiment_model.pkl
│
├── outputs/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── generate_outputs.py
│
├── src/
│   └── yelp_review_intelligence/
│       │
│       ├── api/
│       ├── services/
│       ├── inference/
│       ├── config.py
│       ├── data_preparation.py
│       ├── intelligence.py
│       ├── modeling.py
│       ├── logger.py
│       ├── exceptions.py
│       └── utils.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/eserinanarslan/yelp_review_intelligence.git

cd yelp_review_intelligence
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

Download the Yelp Academic Dataset:

https://www.yelp.com/dataset

Place the raw files under:

```
yelp_datasets/yelp_dataset/
```

Required files:

- yelp_academic_dataset_review.json
- yelp_academic_dataset_business.json
- yelp_academic_dataset_tip.json

---

# Running the Training Pipeline

```
export PYTHONPATH=$PWD/src

python scripts/prepare_data.py --config configs/config.yaml

python scripts/train_model.py --config configs/config.yaml

python scripts/generate_outputs.py --config configs/config.yaml
```

Generated artifacts:

```
data/processed/

outputs/

models/sentiment_model.pkl
```

---

# Running the REST API

```
export PYTHONPATH=$PWD/src

uvicorn yelp_review_intelligence.api.app:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

# Docker

Build

```bash
docker compose build
```

Run

```bash
docker compose up
```

Swagger

```
http://localhost:8000/docs
```

---

# Logging

The application includes centralized logging.

Logs are written to:

```
logs/app.log
```

---

# Exception Handling

Custom exception hierarchy is implemented for:

- Configuration
- Data Preparation
- Model Training
- Model Persistence
- Business Intelligence

---

# Reproducibility

The project is configuration-driven.

All paths, parameters and topic definitions are managed through:

```
configs/config.yaml
```

---

# Leakage Prevention

The pipeline prevents information leakage by:

- Chronological train/test split
- Historical tip aggregation
- No future information during feature engineering
- Separate train and inference pipelines

---

# Future Improvements

Possible future extensions include:

- BERTopic
- Sentence Transformers
- LLM-powered summaries
- RAG-based restaurant assistant
- MLflow model registry
- Kubernetes deployment
- CI/CD pipeline
- Monitoring and drift detection
- Authentication
- Database integration

---

# Tech Stack

- Python
- Pandas
- NumPy
- DuckDB
- Scikit-learn
- FastAPI
- Pydantic
- Docker
- Joblib
- YAML
- Matplotlib

---

# Author

Eser Inan Arslan

Senior Data Scientist | Machine Learning Engineer