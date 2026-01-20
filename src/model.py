"""Model training, evaluation, and prediction with hyperparameter tuning."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from .config import RANDOM_STATE, RISK_PERCENTILE


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to balance classes.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if not SMOTE_AVAILABLE:
        print("Warning: imbalanced-learn not installed, skipping SMOTE")
        return X_train, y_train

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"SMOTE: {len(y_train)} -> {len(y_resampled)} samples")
    print(f"  Class balance: {y_train.mean():.1%} -> {y_resampled.mean():.1%} dropout")

    return X_resampled, y_resampled


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_smote: bool = True,
    tune_hyperparams: bool = True,
) -> Dict[str, Any]:
    """
    Train multiple models with optional SMOTE and hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        use_smote: Whether to apply SMOTE oversampling
        tune_hyperparams: Whether to tune hyperparameters via CV

    Returns:
        Dictionary with model names as keys and fitted models as values
    """
    # Apply SMOTE if requested
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Logistic Regression with tuning
    print("\nTraining Logistic Regression...")
    if tune_hyperparams:
        best_lr, best_lr_score = tune_logistic_regression(X_train, y_train, cv)
        print(f"  Best CV ROC-AUC: {best_lr_score:.3f}")
    else:
        best_lr = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced",
        )
    best_lr.fit(X_train, y_train)
    models["logistic_regression"] = best_lr

    # 2. Random Forest with tuning
    print("\nTraining Random Forest...")
    if tune_hyperparams:
        best_rf, best_rf_score = tune_random_forest(X_train, y_train, cv)
        print(f"  Best CV ROC-AUC: {best_rf_score:.3f}")
    else:
        best_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    best_rf.fit(X_train, y_train)
    models["random_forest"] = best_rf

    # 3. XGBoost with tuning
    if XGBOOST_AVAILABLE:
        print("\nTraining XGBoost...")
        if tune_hyperparams:
            best_xgb, best_xgb_score = tune_xgboost(X_train, y_train, cv)
            print(f"  Best CV ROC-AUC: {best_xgb_score:.3f}")
        else:
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            best_xgb = XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        best_xgb.fit(X_train, y_train)
        models["xgboost"] = best_xgb
    else:
        print("\nXGBoost not available, skipping...")

    return models


def tune_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> Tuple[LogisticRegression, float]:
    """Tune Logistic Regression hyperparameters."""
    best_score = 0
    best_model = None

    param_grid = [
        {"C": 0.01, "penalty": "l2", "solver": "lbfgs"},
        {"C": 0.1, "penalty": "l2", "solver": "lbfgs"},
        {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        {"C": 10.0, "penalty": "l2", "solver": "lbfgs"},
        {"C": 0.1, "penalty": "l1", "solver": "saga"},
        {"C": 1.0, "penalty": "l1", "solver": "saga"},
    ]

    for params in param_grid:
        model = LogisticRegression(
            **params,
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced",
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    return best_model, best_score


def tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> Tuple[RandomForestClassifier, float]:
    """Tune Random Forest hyperparameters."""
    best_score = 0
    best_model = None

    param_grid = [
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 10},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 100, "max_depth": 15, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2},
    ]

    for params in param_grid:
        model = RandomForestClassifier(
            **params,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    return best_model, best_score


def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> Tuple[Any, float]:
    """Tune XGBoost hyperparameters."""
    best_score = 0
    best_model = None

    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    param_grid = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.9},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8},
    ]

    for params in param_grid:
        model = XGBClassifier(
            **params,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    return best_model, best_score


def evaluate_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models and return metrics.

    Args:
        models: Dictionary of model name -> fitted model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with model names as keys and metrics dicts as values
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

    return results


def select_best_model(
    models: Dict[str, Any],
    results: Dict[str, Dict[str, float]],
    metric: str = "roc_auc",
) -> Tuple[str, Any]:
    """
    Select the best model based on a metric.

    Args:
        models: Dictionary of model name -> fitted model
        results: Dictionary of model name -> metrics
        metric: Metric to use for selection (default: roc_auc)

    Returns:
        Tuple of (best model name, best model)
    """
    best_name = max(results.keys(), key=lambda k: results[k][metric])
    return best_name, models[best_name]


def predict_risk_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Predict dropout risk scores (probability of dropout).

    Args:
        model: Fitted model with predict_proba method
        X: Features to predict on

    Returns:
        Array of risk scores (0-1)
    """
    return model.predict_proba(X)[:, 1]


def identify_at_risk(
    df: pd.DataFrame,
    risk_scores: np.ndarray,
    percentile: int = RISK_PERCENTILE,
) -> pd.DataFrame:
    """
    Identify at-risk students based on risk scores.

    Args:
        df: Original student DataFrame (with student_id)
        risk_scores: Array of risk scores
        percentile: Percentile threshold (e.g., 80 = top 20% flagged)

    Returns:
        DataFrame with risk information
    """
    threshold = np.percentile(risk_scores, percentile)

    result = pd.DataFrame({
        "student_id": df["student_id"].values,
        "risk_score": risk_scores,
        "risk_rank": pd.Series(risk_scores).rank(ascending=False).astype(int),
        "is_at_risk": risk_scores >= threshold,
    })

    # Include email and first_name if present (for outreach)
    if "first_name" in df.columns:
        result["first_name"] = df["first_name"].values
    if "email" in df.columns:
        result["email"] = df["email"].values

    # Sort by risk score descending
    result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)

    return result


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    model_type: str,
) -> pd.DataFrame:
    """
    Get feature importance from a model.

    Args:
        model: Fitted model
        feature_names: List of feature names
        model_type: "logistic_regression", "random_forest", or "xgboost"

    Returns:
        DataFrame with feature names and importance scores, sorted by importance
    """
    if model_type == "logistic_regression":
        # For LR, use absolute coefficient values
        importance = np.abs(model.coef_[0])
    elif model_type in ("random_forest", "xgboost"):
        importance = model.feature_importances_
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return result


def get_top_risk_factors(
    student_features: np.ndarray,
    feature_names: List[str],
    model: Any,
    model_type: str,
    top_n: int = 3,
) -> List[str]:
    """
    Get top risk factors for a single student.

    Args:
        student_features: Feature array for one student
        feature_names: List of feature names
        model: Fitted model
        model_type: Type of model
        top_n: Number of top factors to return

    Returns:
        List of top risk factor names
    """
    importance_df = get_feature_importance(model, feature_names, model_type)
    return importance_df.head(top_n)["feature"].tolist()


def save_model(model: Any, preprocessor: Any, filepath: str) -> None:
    """
    Save model and preprocessor to disk.

    Args:
        model: Fitted model
        preprocessor: Fitted preprocessor
        filepath: Path to save to
    """
    joblib.dump({"model": model, "preprocessor": preprocessor}, filepath)


def load_model(filepath: str) -> Tuple[Any, Any]:
    """
    Load model and preprocessor from disk.

    Args:
        filepath: Path to load from

    Returns:
        Tuple of (model, preprocessor)
    """
    data = joblib.load(filepath)
    return data["model"], data["preprocessor"]


def print_evaluation_report(
    results: Dict[str, Dict[str, float]],
    best_model_name: str,
) -> None:
    """
    Print a formatted evaluation report.

    Args:
        results: Dictionary of model name -> metrics
        best_model_name: Name of the selected best model
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    # Sort by ROC AUC
    sorted_names = sorted(results.keys(), key=lambda k: results[k]["roc_auc"], reverse=True)

    for name in sorted_names:
        metrics = results[name]
        marker = " ★ BEST" if name == best_model_name else ""
        print(f"\n{name.upper()}{marker}")
        print("-" * 40)
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.3f}")

    print("\n" + "=" * 60)
    print(f"Selected model: {best_model_name} (highest ROC AUC)")
    print("=" * 60 + "\n")
