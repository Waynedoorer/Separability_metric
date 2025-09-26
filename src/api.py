from src.components import calculate_ks_distance, compute_mmd
import joblib

def tao_score(X, y, model_path="Models/random_forest_model5.pkl", k=5):
    """
    Compute Tao Index score for a dataset.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix.
    y : array-like (n_samples,)
        Binary class labels (0/1).
    model_path : str
        Path to the trained Random Forest regression model.
    k : int
        Number of neighbors for local metrics.

    Returns
    -------
    float
        Predicted Tao Index score.
    """
    rf_model = joblib.load(model_path)

    ks_distance, _ = calculate_ks_distance("distance", X, y, k=k)
    ks_density, _ = calculate_ks_distance("density", X, y, k=k)
    ks_dimension, _ = calculate_ks_distance("dimension", X, y, k=k)
    ks_nlabels, _ = calculate_ks_distance("nlabels", X, y, k=k)

    X0 = X[y == 0]
    X1 = X[y == 1]
    mmd_value = compute_mmd(X0, X1)

    meta = [ks_distance, ks_density, ks_dimension, ks_nlabels, mmd_value]
    return float(rf_model.predict([meta])[0])