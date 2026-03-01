"""
Baron-Kenny style mediation: does Attendance affect Burnout through Engagement?
We fit the path models and report path a (X->M), b (M->Y), c' (direct X->Y), and indirect effect a*b.
"""
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data = pd.read_csv("data/train_features.csv")

    gender_cols = [c for c in data.columns if c.startswith("gender_")]
    gender_term = f" + {gender_cols[0]}" if gender_cols else ""

    # Path a: X (attendance) -> M (engagement_index)
    mediator_formula = f"engagement_index ~ attendance + age{gender_term}"
    mediator_model = sm.OLS.from_formula(mediator_formula, data=data).fit()
    print("=== Model 1: Mediator (Engagement) ~ Attendance + covariates ===")
    print(mediator_model.summary())

    # Paths b and c': Y (burnout) ~ X + M + covariates
    outcome_formula = f"burnout_score ~ attendance + engagement_index + age{gender_term}"
    outcome_model = sm.OLS.from_formula(outcome_formula, data=data).fit()
    print("\n=== Model 2: Outcome (Burnout) ~ Attendance + Engagement + covariates ===")
    print(outcome_model.summary())

    # Total effect c: Y ~ X (no M)
    total_formula = f"burnout_score ~ attendance + age{gender_term}"
    total_model = sm.OLS.from_formula(total_formula, data=data).fit()

    # Baron-Kenny mediation summary
    a = mediator_model.params["attendance"]
    b = outcome_model.params["engagement_index"]
    c_prime = outcome_model.params["attendance"]
    c_total = total_model.params["attendance"]
    indirect = a * b

    print("\n=== Mediation (Baron-Kenny) ===")
    print(f"  Path a (Attendance -> Engagement):     {a:.4f}")
    print(f"  Path b (Engagement -> Burnout):        {b:.4f}")
    print(f"  Direct effect c' (Attendance -> Burnout): {c_prime:.4f}")
    print(f"  Total effect c (Attendance -> Burnout):  {c_total:.4f}")
    print(f"  Indirect effect (a*b):                  {indirect:.4f}")
    if abs(c_total) > 1e-8:
        pct_mediated = 100 * indirect / c_total
        print(f"  Proportion mediated (indirect/total): {pct_mediated:.1f}%")


if __name__ == "__main__":
    main()

