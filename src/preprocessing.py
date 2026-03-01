import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def main() -> None:
    df = pd.read_csv("data/student_data.csv")

    # Basic missing-value handling (dataset is synthetic but this keeps things robust)
    df.fillna(df.median(numeric_only=True), inplace=True)
    if df.isna().any().any():
        df.fillna(df.mode().iloc[0], inplace=True)

    # Encode categoricals
    df = pd.get_dummies(df, columns=["gender", "income_level", "nationality"], drop_first=True)

    # Label encode burnout_level
    le = LabelEncoder()
    df["burnout_level_code"] = le.fit_transform(df["burnout_level"])

    # Ensure dropout_status is int
    df["dropout_status"] = df["dropout_status"].astype(int)

    # Standardize numeric features
    num_feats = [
        "age",
        "GPA",
        "attendance",
        "course_completion",
        "assignment_delays",
        "LMS_logins",
        "forum_posts",
        "night_study_hours",
        "engagement_score",
        "sentiment_score",
        "motivation",
        "emotional_exhaustion",
        "depersonalization",
        "personal_accomplishment",
        "autonomy",
        "competence",
        "relatedness",
        "O",
        "C",
        "E",
        "A",
        "N",
    ]

    scaler = StandardScaler()
    df[num_feats] = scaler.fit_transform(df[num_feats])

    # Train / test split stratified on burnout_level_code
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["burnout_level_code"],
    )

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


if __name__ == "__main__":
    main()

