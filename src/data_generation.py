import numpy as np
import pandas as pd


def generate_student_data(n_students: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic student dataset with demographic, academic, behavioural and
    psychological variables, plus burnout and dropout targets.
    """
    rng = np.random.default_rng(random_state)

    data = pd.DataFrame(
        {
            "age": rng.integers(18, 30, size=n_students),
            "gender": rng.choice(["M", "F"], size=n_students),
            "income_level": rng.choice(["Low", "Medium", "High"], size=n_students, p=[0.3, 0.5, 0.2]),
            "nationality": rng.choice(["US", "IN", "CN", "UK", "DE"], size=n_students),
        }
    )

    # Academic features
    data["GPA"] = np.round(rng.normal(3.0, 0.5, size=n_students), 2)
    data["GPA"] = data["GPA"].clip(0.0, 4.0)

    data["attendance"] = np.clip(rng.normal(80, 10, size=n_students), 0, 100)
    data["course_completion"] = rng.integers(0, 101, size=n_students)
    data["assignment_delays"] = rng.poisson(2, size=n_students)

    # Behavioural features
    base_logins = rng.poisson(30, size=n_students)
    # Slightly fewer logins for older students
    lms_raw = base_logins - 0.5 * (data["age"].values - data["age"].mean())
    data["LMS_logins"] = np.maximum(lms_raw, 0)
    data["forum_posts"] = rng.poisson(5, size=n_students)
    data["night_study_hours"] = np.clip(rng.normal(10, 3, size=n_students), 0, None)
    data["engagement_score"] = np.clip(rng.normal(70, 15, size=n_students), 0, 100)
    data["sentiment_score"] = rng.uniform(-1, 1, size=n_students)

    # Psychological / personality features
    data["motivation"] = np.clip(rng.normal(60, 20, size=n_students), 0, 100)

    # Burnout subscales 0–50
    data["emotional_exhaustion"] = rng.integers(0, 51, size=n_students)
    data["depersonalization"] = rng.integers(0, 51, size=n_students)
    data["personal_accomplishment"] = rng.integers(0, 51, size=n_students)

    # Self-Determination Theory traits (1–7 Likert)
    data["autonomy"] = rng.integers(1, 7, size=n_students)
    data["competence"] = rng.integers(1, 7, size=n_students)
    data["relatedness"] = rng.integers(1, 7, size=n_students)

    # Big Five traits 0–100
    for trait in ["O", "C", "E", "A", "N"]:
        data[trait] = rng.integers(0, 101, size=n_students)

    # Overall burnout_score and burnout_level
    data["burnout_score"] = (
        data["emotional_exhaustion"] + data["depersonalization"] - data["personal_accomplishment"]
    )
    data["burnout_score"] = data["burnout_score"].clip(lower=0, upper=100)
    data["burnout_level"] = pd.cut(
        data["burnout_score"],
        bins=[-1, 30, 60, 100],
        labels=["Low", "Medium", "High"],
    )

    # Dropout probability: higher for older, low GPA, high burnout, low engagement
    burnout_term = np.interp(data["burnout_score"], [0, 100], [0.0, 1.5])
    gpa_term = np.interp(data["GPA"], [0, 4], [1.0, -1.0])
    age_term = np.interp(data["age"], [18, 30], [0.0, 0.5])
    engagement_term = np.interp(data["engagement_score"], [0, 100], [0.8, -0.8])

    logit_p = -1.0 + burnout_term + gpa_term + age_term + engagement_term
    prob_dropout = 1 / (1 + np.exp(-logit_p))
    prob_dropout = np.clip(prob_dropout, 0.05, 0.9)

    data["dropout_status"] = rng.binomial(1, prob_dropout)

    # Time-to-dropout in weeks: shorter for high burnout, longer for low burnout
    base_scale = np.interp(data["burnout_score"], [0, 100], [60, 20])
    base_time = rng.exponential(scale=base_scale)

    # Add some random administrative censoring for non-dropouts
    censor_time = rng.exponential(scale=80, size=n_students)
    observed_time = np.where(data["dropout_status"] == 1, base_time, np.minimum(base_time, censor_time))

    data["time_to_dropout"] = np.maximum(observed_time.astype(int), 1)

    return data


def main() -> None:
    df = generate_student_data()
    output_path = "data/student_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path} with shape {df.shape}")


if __name__ == "__main__":
    main()

