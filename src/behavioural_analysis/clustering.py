import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def main() -> None:
    os.makedirs("plots", exist_ok=True)
    data = pd.read_csv("data/train_features.csv")
    X = data[["engagement_index", "motivation_proxy"]].dropna()

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    data = data.loc[X.index].copy()
    data["cluster"] = labels

    out_csv = "data/clusters.csv"
    data.to_csv(out_csv, index=False)
    print(f"Saved clustered data with labels to {out_csv}")

    # 2D scatter plot
    plt.figure(figsize=(7, 5))
    for c in sorted(data["cluster"].unique()):
        subset = data[data["cluster"] == c]
        plt.scatter(
            subset["engagement_index"],
            subset["motivation_proxy"],
            label=f"Cluster {c}",
            alpha=0.7,
        )
    plt.xlabel("Engagement Index")
    plt.ylabel("Motivation Proxy")
    plt.title("Student Clusters")
    plt.legend()
    plt.tight_layout()

    out_plot = "plots/clusters_engagement_motivation.png"
    plt.savefig(out_plot)
    plt.close()
    print(f"Saved cluster scatter plot to {out_plot}")


if __name__ == "__main__":
    main()

