import argparse
from yelp_review_intelligence.config import ProjectConfig
from yelp_review_intelligence.data_preparation import YelpDataPreparation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    pipeline = YelpDataPreparation(config)
    train_df, test_df = pipeline.run()
    train_df.to_pickle(config.processed_dir / "train_model_df.pkl")
    test_df.to_pickle(config.processed_dir / "test_model_df.pkl")
    print(f"Saved train dataframe: {train_df.shape}")
    print(f"Saved test dataframe: {test_df.shape}")


if __name__ == "__main__":
    main()
