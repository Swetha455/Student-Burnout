"""
Run the full behavioural analytics pipeline in order.
Execute from project root:
  python run_pipeline.py              # run all steps
  python run_pipeline.py --from 5     # run from step 5 to the end
  python run_pipeline.py --list       # show step numbers and names
"""
import os
import subprocess
import sys

# Project root = directory containing this script
ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("src/data_generation.py", "Generating synthetic dataset"),
    ("src/preprocessing.py", "Preprocessing and train/test split"),
    ("src/feature_engineering.py", "Feature engineering"),
    ("src/model_training.py", "Training models"),
    ("src/explainability.py", "SHAP explainability"),
    ("src/behavioural_analysis/mediation_analysis.py", "Mediation analysis"),
    ("src/behavioural_analysis/moderation_analysis.py", "Moderation analysis"),
    ("src/behavioural_analysis/causal_graphs.py", "Causal DAG"),
    ("src/behavioural_analysis/anomaly_detection.py", "Anomaly detection"),
    ("src/behavioural_analysis/clustering.py", "Clustering"),
    ("src/behavioural_analysis/factor_analysis.py", "Factor analysis"),
    ("src/behavioural_analysis/survival_analysis.py", "Survival analysis"),
]


def main():
    start_from = 1
    if "--list" in sys.argv or "-l" in sys.argv:
        print("Pipeline steps (use: python run_pipeline.py --from N):")
        for i, (script, label) in enumerate(STEPS, 1):
            print(f"  {i:2}. {label}")
        return
    for i, arg in enumerate(sys.argv):
        if arg in ("--from", "-f") and i + 1 < len(sys.argv):
            try:
                start_from = max(1, int(sys.argv[i + 1]))
            except ValueError:
                pass
            break

    for i, (script, label) in enumerate(STEPS, 1):
        if i < start_from:
            continue
        print(f"\n--- Step {i}/{len(STEPS)}: {label} ---")
        result = subprocess.run([sys.executable, script], cwd=ROOT)
        if result.returncode != 0:
            print(f"Pipeline failed at step {i}: {script}")
            sys.exit(result.returncode)
    print("\n--- Pipeline finished successfully ---")


if __name__ == "__main__":
    main()
