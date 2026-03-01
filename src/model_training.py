import os

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from numpy.linalg import LinAlgError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def train_models() -> None:
    train = pd.read_csv("data/train_features.csv")
    test = pd.read_csv("data/test_features.csv")

    # Define feature set (exclude targets and non-feature columns)
    drop_cols = [
        "burnout_level",
        "burnout_level_code",
        "dropout_status",
        "time_to_dropout",
    ]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    y_train_burnout = train["burnout_level_code"]
    y_test_burnout = test["burnout_level_code"]

    y_train_dropout = train["dropout_status"]
    y_test_dropout = test["dropout_status"]

    os.makedirs("models", exist_ok=True)

    # ------------------------
    # Random Forest: burnout level
    # ------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train_burnout)

    pred_burnout = rf.predict(X_test)
    prob_burnout = rf.predict_proba(X_test)

    f1_burnout = f1_score(y_test_burnout, pred_burnout, average="weighted")

    # Multi-class ROC-AUC using one-vs-one
    y_test_burnout_onehot = pd.get_dummies(y_test_burnout)
    roc_burnout = roc_auc_score(
        y_test_burnout_onehot,
        prob_burnout,
        multi_class="ovo",
    )

    cm_burnout = confusion_matrix(y_test_burnout, pred_burnout)

    print("=== Burnout Level (Random Forest) ===")
    print(f"F1 (weighted): {f1_burnout:.3f}")
    print(f"ROC-AUC (ovo): {roc_burnout:.3f}")
    print("Confusion matrix:")
    print(cm_burnout)

    joblib.dump(rf, "models/rf_burnout.pkl")

    # ------------------------
    # Logistic Regression: dropout
    # ------------------------
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_train, y_train_dropout)

    pred_dropout = lr.predict(X_test)
    prob_dropout = lr.predict_proba(X_test)[:, 1]

    f1_drop = f1_score(y_test_dropout, pred_dropout)
    roc_drop = roc_auc_score(y_test_dropout, prob_dropout)
    cm_drop = confusion_matrix(y_test_dropout, pred_dropout)

    print("\n=== Dropout Status (Logistic Regression) ===")
    print(f"F1: {f1_drop:.3f}")
    print(f"ROC-AUC: {roc_drop:.3f}")
    print("Confusion matrix:")
    print(cm_drop)

    joblib.dump(lr, "models/logreg_dropout.pkl")

    # ------------------------
    # Cox Proportional Hazards: time-to-dropout
    # ------------------------
    surv_df = train[["time_to_dropout", "dropout_status"] + feature_cols].copy()

    # Drop constant columns to avoid singular matrix
    cox_feats = [c for c in feature_cols if c in surv_df.columns and surv_df[c].nunique() > 1]
    surv_df = surv_df[["time_to_dropout", "dropout_status"] + cox_feats].copy()

    # Use L2 penalizer to stabilize fit when design matrix is ill-conditioned
    for penalizer in [0.5, 2.0, 5.0]:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(
                surv_df,
                duration_col="time_to_dropout",
                event_col="dropout_status",
                show_progress=False,
            )
            break
        except (LinAlgError, ValueError) as e:
            if penalizer == 5.0:
                print(f"\nCox model skipped (singular/ill-conditioned matrix): {e}")
                return
            continue

    print("\n=== Cox Proportional Hazards (Time to Dropout) ===")
    cph.print_summary()

    joblib.dump(cph, "models/cox_dropout.pkl")


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()

