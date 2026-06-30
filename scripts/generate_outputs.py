import argparse
import pandas as pd
from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.intelligence import ReviewIntelligenceGenerator
from yelp_review_intelligence.utils import save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--business-id", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    predictions_df = pd.read_pickle(config.processed_dir / "test_predictions.pkl")
    generator = ReviewIntelligenceGenerator(config)

    topic_level_df = generator.create_topic_level_df(predictions_df)
    business_insights = generator.create_business_insights(topic_level_df)
    dashboard_output, overall_sentiment, topic_summary, representative_reviews = generator.create_dashboard_output(
        predictions_df, topic_level_df, business_id=args.business_id
    )

    topic_level_df.to_pickle(config.processed_dir / "topic_level_predictions.pkl")
    business_insights.to_csv(config.output_dir / "business_topic_insights.csv", index=False)
    overall_sentiment.to_csv(config.output_dir / "overall_sentiment_example_business.csv", index=False)
    topic_summary.to_csv(config.output_dir / "topic_summary_example_business.csv", index=False)
    representative_reviews.to_csv(config.output_dir / "representative_reviews_example_business.csv", index=False)
    save_json(dashboard_output, config.output_dir / "dashboard_output_example_business.json")
    print("Dashboard output generated.")


if __name__ == "__main__":
    main()
