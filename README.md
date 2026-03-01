## Advanced Behavioural Analytics for Early Detection of Student Burnout and Dropout Risk

This project implements an end‑to‑end behavioural analytics pipeline to **predict student burnout and dropout risk**, explain key behavioural drivers, and surface insights through an **interactive Streamlit dashboard**.

The system uses a mixed synthetic / real‑inspired dataset (~3,000 students) with:

- **Demographics**: age, gender, income_level, nationality  
- **Academic**: GPA, attendance, course_completion, assignment_delays  
- **Behavioural**: LMS_logins, forum_posts, night_study_hours, engagement_score, sentiment_score  
- **Psychological**: burnout_score and subscales, motivation, SDT traits (autonomy, competence, relatedness), Big Five traits (O, C, E, A, N)  
- **Targets**: dropout_status (0/1), burnout_level (Low/Medium/High), time_to_dropout (weeks, with censoring)

### Dataset & Generation

- **Script**: `src/data_generation.py`  
- **Output**: `data/student_data.csv` (~3,000 rows)  
- **Logic**:
  - Simulates realistic distributions inspired by public datasets (e.g. UCI / Open University).
  - Introduces **behavioural–academic–psychological links** (e.g. low GPA and low engagement → higher burnout and dropout risk).
  - Derives `burnout_score` and categorical `burnout_level` from burnout subscales.
  - Generates **dropout_status** and **time_to_dropout** with higher hazard for high burnout.

### Preprocessing

- **Script**: `src/preprocessing.py`  
- **Steps**:
  - Handles **missing values** (numeric → median, categorical → mode).
  - **Encodes categoricals** (gender, income_level, nationality) using one‑hot encoding.
  - **Standardizes numeric features** with `StandardScaler`.
  - Creates **train / test split** (80/20) stratified on `burnout_level_code`.  
- **Outputs**: `data/train.csv`, `data/test.csv`.

### Feature Engineering

- **Script**: `src/feature_engineering.py`  
- **Key composite features**:
  - **Engagement Index**: combines LMS usage and participation.  
  - **Procrastination Score**: higher for many assignment_delays and lower conscientiousness (C).  
  - **Behaviour Drift**: discrepancy between LMS_logins and attendance.  
  - **Motivation Proxy**: mean of autonomy, competence, relatedness.  
  - **Sentiment‑Adjusted Stress**: emotional_exhaustion amplified by negative sentiment.  
- **Outputs**: `data/train_features.csv`, `data/test_features.csv`.

### Modelling

- **Script**: `src/model_training.py`  
- **Models**:
  - **Random Forest classifier** for burnout level (`burnout_level_code`).  
  - **Logistic Regression** for dropout (`dropout_status`).  
  - **Cox Proportional Hazards** model for **time‑to‑dropout** (lifelines).  
- **Evaluation**:
  - F1‑score, ROC‑AUC, confusion matrices for classifiers.
  - Concordance index and coefficient summary for Cox model.  
- **Saved models**: `models/rf_burnout.pkl`, `models/logreg_dropout.pkl`, `models/cox_dropout.pkl`.

### Explainability

- **Script**: `src/explainability.py`  
- **Tools**:
  - SHAP TreeExplainer for the Random Forest burnout model.
  - Global feature importance (mean |SHAP|) to identify **top behavioural drivers** (e.g. engagement_index, procrastination_score).  
- **Outputs**:
  - Console summary of top features.
  - Optional plots saved to `plots/` (e.g. SHAP summary bar plot).

### Behavioural Analyses

Located in `src/behavioural_analysis/`:

- **Mediation analysis** (`mediation_analysis.py`):  
  - Tests whether **engagement_index mediates** the relationship between **attendance** and **burnout_score** using statsmodels’ mediation framework.
- **Moderation analysis** (`moderation_analysis.py`):  
  - Tests whether **sentiment_score moderates** the impact of **procrastination_score** on **burnout_score** via interaction terms.
- **Causal graphs** (`causal_graphs.py`):  
  - Defines a conceptual **causal DAG** of attendance, engagement, burnout, and dropout using NetworkX and (optionally) DoWhy/PyWhy.
- **Anomaly detection** (`anomaly_detection.py`):  
  - Uses **Isolation Forest** to flag students with unusual behavioural patterns (e.g. extremely low engagement but high grades).
- **Clustering** (`clustering.py`):  
  - Applies **K‑Means** on engagement / motivation features to discover **student personas**.
- **Factor analysis** (`factor_analysis.py`):  
  - Uses **Exploratory Factor Analysis** on burnout subscales to confirm latent burnout dimensions.
- **Survival analysis** (`survival_analysis.py`):  
  - Plots **Kaplan–Meier survival curves** and supports further inspection of dropout over time.

Each script prints textual summaries and may save plots into `plots/`.

### Dashboard

- **App**: `dashboard/app.py` (Streamlit)  
- **Features**:
  - Header with **key statistics** (dropout rate, average burnout).  
  - **Burnout distribution and predictions** (bar charts).  
  - **Dropout model performance** (ROC‑like diagnostics / probability distributions).  
  - **SHAP feature importance** visualisation.  
  - **Student clusters** scatterplot (engagement vs motivation).  
  - **Anomaly table** with flagged students.  
  - **Survival curves** (Kaplan–Meier).  
  - Optional mediation / moderation diagrams and other charts using Streamlit layouts and Altair.

### Project Structure

- **`src/`**: core Python modules  
  - `data_generation.py`  
  - `preprocessing.py`  
  - `feature_engineering.py`  
  - `model_training.py`  
  - `explainability.py`  
  - `behavioural_analysis/` (mediation, moderation, DAGs, clustering, anomalies, factor & survival analysis)
- **`dashboard/`**: Streamlit app (`app.py`)  
- **`data/`**: generated CSVs (raw, train/test, engineered)  
- **`models/`**: saved `.pkl` model files  
- **`plots/`**: saved figures (SHAP, KM curves, clusters, etc.)

### How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full pipeline**

   From the project root:

   ```bash
   python run_pipeline.py
   ```

   To **start from a specific step** (so you don’t re-run from the beginning):

   ```bash
   python run_pipeline.py --from 5
   ```

   Step 5 = explainability; step 4 = model training; etc. To see step numbers:

   ```bash
   python run_pipeline.py --list
   ```

   If any step fails, the script stops and reports which step failed.

3. **Or run steps individually**

   You can still run each script one by one if you prefer (e.g. to re-run only one step):

   ```bash
   python src/data_generation.py
   python src/preprocessing.py
   python src/feature_engineering.py
   python src/model_training.py
   python src/explainability.py
   python src/behavioural_analysis/mediation_analysis.py
   python src/behavioural_analysis/moderation_analysis.py
   python src/behavioural_analysis/causal_graphs.py
   python src/behavioural_analysis/anomaly_detection.py
   python src/behavioural_analysis/clustering.py
   python src/behavioural_analysis/factor_analysis.py
   python src/behavioural_analysis/survival_analysis.py
   ```

4. **Launch dashboard**

   ```bash
   streamlit run dashboard/app.py
   ```

   Then open `http://localhost:8501` in your browser.

### Notes & References

- **Techniques used**:
  - Machine learning for **dropout / burnout prediction** (Random Forest, Logistic Regression).  
  - **Explainable AI** with SHAP for feature attributions.  
  - **Survival analysis** (Kaplan–Meier, Cox proportional hazards) for time‑to‑dropout.  
  - **Mediation / moderation analysis** for behavioural mechanisms.  
  - **Isolation Forest** for anomaly detection.  
  - **Factor analysis** to validate burnout constructs.
- These techniques are widely used in **educational data mining**, **learning analytics**, and **behavioural science** literature; you can plug in real institutional data (with appropriate ethics and governance) by replacing the synthetic data generation step.

