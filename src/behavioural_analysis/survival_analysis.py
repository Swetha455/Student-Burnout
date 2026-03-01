import os

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter


def main() -> None:
    os.makedirs("plots", exist_ok=True)
    data = pd.read_csv("data/train_features.csv")
    T = data["time_to_dropout"]
    E = data["dropout_status"]

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, label="All Students")

    plt.figure(figsize=(7, 5))
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve (Dropout Over Time)")
    plt.xlabel("Weeks since enrollment")
    plt.ylabel("Survival probability")
    plt.tight_layout()

    out_path = "plots/km_overall.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Kaplan-Meier curve to {out_path}")


if __name__ == "__main__":
    main()

