# Yelp Review Intelligence

End-to-end NLP case study for a fictive stakeholder: a restaurant owner or multi-location restaurant operations team.

The goal is not only to classify reviews. The goal is to transform unstructured customer review text into structured, actionable business insights:

- sentiment
- complaint topics
- severity
- representative reviews
- recommended actions
- dashboard-ready tables

## Structure

```text
yelp_review_intelligence_project/
├── configs/config.yaml
├── requirements.txt
├── scripts/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── generate_outputs.py
├── src/yelp_review_intelligence/
│   ├── config.py
│   ├── data_preparation.py
│   ├── modeling.py
│   ├── intelligence.py
│   └── utils.py
├── data/processed/
├── models/
└── outputs/
```

## Setup

```bash
pip install -r requirements.txt
```

Place the Yelp raw files under:

```text
yelp_datasets/yelp_dataset/
```

Expected files:

- yelp_academic_dataset_review.json
- yelp_academic_dataset_business.json
- yelp_academic_dataset_tip.json

## Run

```bash
export PYTHONPATH=$PWD/src
python scripts/prepare_data.py --config configs/config.yaml
python scripts/train_model.py --config configs/config.yaml
python scripts/generate_outputs.py --config configs/config.yaml
```

## Leakage Control

The pipeline uses a chronological train/test split. Tip context is aggregated only up to the training cut-off date and then joined to both train and test sets. This prevents future tips from leaking into training or evaluation.

## Why OOP?

The notebooks are now converted into reusable components:

- `YelpDataPreparation`: raw data reading, leakage-safe split, feature engineering
- `SentimentModelTrainer`: ablation, grid search, evaluation, feature importance
- `ReviewIntelligenceGenerator`: topic detection, severity, business-level dashboard outputs
