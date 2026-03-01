import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite behavioural / psychological indices to a preprocessed dataframe.
    Assumes preprocessing has already standardised core numeric variables.
    """
    fe_df = df.copy()

    # Engagement index: simple weighted combination of activity metrics
    fe_df["engagement_index"] = (
        fe_df["LMS_logins"] + 2 * fe_df["forum_posts"] + fe_df["engagement_score"]
    ) / 4.0

    # Procrastination score: late assignments penalised more when conscientiousness (C) is low
    fe_df["procrastination_score"] = fe_df["assignment_delays"] * (1 - fe_df["C"] / 100.0)

    # Behaviour drift: discrepancy between LMS logins and attendance
    fe_df["behaviour_drift"] = (fe_df["LMS_logins"] - fe_df["attendance"]).abs()

    # Motivation proxy: SDT needs
    fe_df["motivation_proxy"] = (
        fe_df["autonomy"] + fe_df["competence"] + fe_df["relatedness"]
    ) / 3.0

    # Sentiment-adjusted stress
    fe_df["sentiment_stress"] = fe_df["emotional_exhaustion"] * (1 - fe_df["sentiment_score"])

    return fe_df


def main() -> None:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train_fe = add_engineered_features(train)
    test_fe = add_engineered_features(test)

    train_fe.to_csv("data/train_features.csv", index=False)
    test_fe.to_csv("data/test_features.csv", index=False)

    print(f"Train with features: {train_fe.shape}, Test with features: {test_fe.shape}")


if __name__ == "__main__":
    main()

