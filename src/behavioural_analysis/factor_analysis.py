"""
Exploratory factor analysis for burnout subscales using sklearn PCA and correlation matrix.
Avoids factor_analyzer to prevent compatibility issues with newer scikit-learn.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def bartlett_sphericity(X: np.ndarray) -> tuple[float, float]:
    """Bartlett's test of sphericity: H0 that correlation matrix is identity. Returns (chi_sq, p_value)."""
    n, p = X.shape
    R = np.corrcoef(X.T)
    det_r = np.linalg.det(R)
    det_r = max(det_r, 1e-300)
    chi_sq = -np.log(det_r) * (n - 1 - (2 * p + 5) / 6)
    df = p * (p - 1) // 2
    p_value = float(chi2.sf(chi_sq, df))
    return float(chi_sq), p_value


def main() -> None:
    data = pd.read_csv("data/train_features.csv")
    cols = ["emotional_exhaustion", "depersonalization", "personal_accomplishment"]
    burnout_sub = data[cols].dropna()

    X = burnout_sub.values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    chi_sq, p_value = bartlett_sphericity(X_std)
    print(f"Bartlett's test chi-square: {chi_sq:.3f}, p-value: {p_value:.5f}")

    # PCA as factor-like structure (2 components)
    pca = PCA(n_components=2)
    pca.fit(X_std)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    print("Factor loadings (PCA-based, 2 components):")
    for i, col in enumerate(cols):
        print(f"  {col}: [{loadings[i, 0]:.3f}, {loadings[i, 1]:.3f}]")
    print(f"Variance explained: {pca.explained_variance_ratio_.round(3)}")


if __name__ == "__main__":
    main()

