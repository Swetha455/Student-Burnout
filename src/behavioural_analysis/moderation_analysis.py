import pandas as pd
from statsmodels.formula.api import ols


def main() -> None:
    data = pd.read_csv("data/train_features.csv")

    # Interaction term: procrastination x sentiment
    data["prog_x_sent"] = data["procrastination_score"] * data["sentiment_score"]

    model = ols(
        "burnout_score ~ procrastination_score + sentiment_score + prog_x_sent",
        data=data,
    ).fit()

    print(model.summary())


if __name__ == "__main__":
    main()

