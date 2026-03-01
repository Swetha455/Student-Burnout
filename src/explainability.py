import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_for_burnout_model() -> None:
    model_path = "models/rf_burnout.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Train models first with src/model_training.py."
        )

    rf = joblib.load(model_path)
    train = pd.read_csv("data/train_features.csv")

    drop_cols = [
        "burnout_level",
        "burnout_level_code",
        "dropout_status",
        "time_to_dropout",
    ]
    feature_cols = [c for c in train.columns if c not in drop_cols]
    X = train[feature_cols]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # Handle binary vs multi-class outputs
    if isinstance(shap_values, list):
        # shape: (n_classes, n_samples, n_features)
        sv = np.stack(shap_values, axis=0)
        mean_abs_shap = np.mean(np.abs(sv), axis=(0, 1))
    else:
        # shape: (n_samples, n_features)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Ensure 1D array of scalars so sort key never gets an array
    mean_abs_shap = np.asarray(mean_abs_shap).ravel()
    feature_importance = sorted(
        zip(mean_abs_shap, feature_cols),
        key=lambda x: float(x[0]),
        reverse=True,
    )

    top_k = 10
    top_features = feature_importance[:top_k]
    print("Top features for burnout prediction (by mean |SHAP|):")
    for val, name in top_features:
        print(f"{name}: {val:.4f}")

    os.makedirs("plots", exist_ok=True)

    # Simple horizontal bar plot of top features
    vals = [float(v) for v, _ in top_features][::-1]
    names = [n for _, n in top_features][::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(names, vals)
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top Features for Burnout Prediction (Random Forest)")
    plt.tight_layout()
    out_path = "plots/shap_burnout_feature_importance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved SHAP feature importance plot to {out_path}")


def main() -> None:
    compute_shap_for_burnout_model()


if __name__ == "__main__":
    main()

