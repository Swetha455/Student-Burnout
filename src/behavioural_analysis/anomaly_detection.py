import pandas as pd
from sklearn.ensemble import IsolationForest


def main() -> None:
    data = pd.read_csv("data/train_features.csv")

    features = ["LMS_logins", "forum_posts", "night_study_hours", "attendance"]
    features = [f for f in features if f in data.columns]

    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(data[features])

    data["anomaly_score"] = iso.decision_function(data[features])
    data["is_anomaly"] = iso.predict(data[features]) == -1

    outliers = data[data["is_anomaly"]]
    print(f"Detected {len(outliers)} anomalous student records.")

    out_path = "data/anomalies.csv"
    outliers.to_csv(out_path, index=False)
    print(f"Saved anomalies to {out_path}")


if __name__ == "__main__":
    main()

