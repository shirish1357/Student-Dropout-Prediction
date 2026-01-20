"""Data preprocessing and feature engineering."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from .config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)


class StudentPreprocessor:
    """Preprocessor for student dropout prediction data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names: list = []
        self._fitted = False

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw data.

        Basic derived features:
        - completion_rate: credits_completed / credits_attempted
        - submission_rate: assignments_submitted / assignments_total
        - late_rate: late_submissions / assignments_submitted
        - days_enrolled: days since enrollment_date

        Advanced features:
        - engagement_score: composite of LMS logins, submissions, attendance
        - gpa_trend_proxy: GPA relative to failed courses (higher = recovering)
        - academic_momentum: completion_rate * gpa (compound success indicator)
        - risk_composite: weighted combination of risk factors
        - low_gpa_flag: binary flag for GPA < 2.0
        - low_attendance_flag: binary flag for attendance < 70%
        - disengaged_flag: binary flag for very low LMS activity

        Args:
            df: DataFrame with raw student data

        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()

        # Basic derived features
        df["completion_rate"] = np.where(
            df["credits_attempted"] > 0,
            df["credits_completed"] / df["credits_attempted"],
            0,
        )

        df["submission_rate"] = np.where(
            df["assignments_total"] > 0,
            df["assignments_submitted"] / df["assignments_total"],
            0,
        )

        df["late_rate"] = np.where(
            df["assignments_submitted"] > 0,
            df["late_submissions"] / df["assignments_submitted"],
            0,
        )

        today = pd.Timestamp(datetime.now())
        df["days_enrolled"] = (today - pd.to_datetime(df["enrollment_date"])).dt.days

        # Advanced features

        # Engagement score: normalized composite of engagement metrics
        lms_norm = df["lms_logins_last_30d"] / 30  # Normalize to ~1
        submission_norm = df["submission_rate"]
        attendance_norm = df["attendance_rate"] / 100
        df["engagement_score"] = (lms_norm + submission_norm + attendance_norm) / 3

        # GPA trend proxy: if GPA is decent despite failed courses, student may be recovering
        df["gpa_trend_proxy"] = df["gpa"] / (df["failed_courses"] + 1)

        # Academic momentum: compound success indicator
        df["academic_momentum"] = df["completion_rate"] * df["gpa"]

        # Risk composite: weighted sum of normalized risk factors
        gpa_risk = (4.0 - df["gpa"]) / 4.0  # Higher = more risk
        attendance_risk = (100 - df["attendance_rate"]) / 100
        submission_risk = 1 - df["submission_rate"]
        late_risk = df["late_rate"]
        df["risk_composite"] = (
            0.35 * gpa_risk +
            0.25 * attendance_risk +
            0.25 * submission_risk +
            0.15 * late_risk
        )

        # Binary risk flags (capture threshold effects)
        df["low_gpa_flag"] = (df["gpa"] < 2.0).astype(int)
        df["low_attendance_flag"] = (df["attendance_rate"] < 70).astype(int)
        df["disengaged_flag"] = (df["lms_logins_last_30d"] < 5).astype(int)

        # Interaction terms
        df["gpa_x_attendance"] = df["gpa"] * df["attendance_rate"] / 100
        df["gpa_x_engagement"] = df["gpa"] * df["engagement_score"]

        return df

    def fit(self, df: pd.DataFrame) -> "StudentPreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            df: Training DataFrame (should include derived features)

        Returns:
            Self for chaining
        """
        # All derived features (basic + advanced)
        derived_features = [
            "completion_rate",
            "submission_rate",
            "late_rate",
            "days_enrolled",
            "engagement_score",
            "gpa_trend_proxy",
            "academic_momentum",
            "risk_composite",
            "gpa_x_attendance",
            "gpa_x_engagement",
        ]

        # Binary flag features (don't scale these)
        self.flag_features = [
            "low_gpa_flag",
            "low_attendance_flag",
            "disengaged_flag",
        ]

        # Fit scaler on numeric features + derived features (excluding flags)
        self.numeric_cols = NUMERIC_FEATURES + derived_features
        self.scaler.fit(df[self.numeric_cols])

        # Fit encoder on categorical features
        self.encoder.fit(df[CATEGORICAL_FEATURES])

        # Store feature names
        self.feature_names = (
            self.numeric_cols
            + list(self.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
            + BOOLEAN_FEATURES
            + self.flag_features
        )

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.

        Args:
            df: DataFrame to transform (should include derived features)

        Returns:
            NumPy array of transformed features
        """
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Scale numeric features
        numeric_scaled = self.scaler.transform(df[self.numeric_cols])

        # Encode categorical features
        categorical_encoded = self.encoder.transform(df[CATEGORICAL_FEATURES])

        # Boolean features (convert to 0/1)
        boolean_values = df[BOOLEAN_FEATURES].astype(int).values

        # Binary flag features
        flag_values = df[self.flag_features].values

        # Combine all features
        X = np.hstack([numeric_scaled, categorical_encoded, boolean_values, flag_values])

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list:
        """Get the names of all features after transformation."""
        return self.feature_names


def prepare_training_data(
    df: pd.DataFrame,
    preprocessor: Optional[StudentPreprocessor] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StudentPreprocessor]:
    """
    Prepare data for model training.

    Args:
        df: DataFrame with student data (must include target column)
        preprocessor: Optional pre-fitted preprocessor
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Create derived features
    if preprocessor is None:
        preprocessor = StudentPreprocessor()

    df_features = preprocessor.create_features(df)

    # Extract target
    y = df_features[TARGET_COLUMN].astype(int).values

    # Split first (to avoid data leakage in scaling)
    indices = np.arange(len(df_features))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    df_train = df_features.iloc[train_idx]
    df_test = df_features.iloc[test_idx]

    # Fit on training data only
    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)

    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test, preprocessor


def prepare_prediction_data(
    df: pd.DataFrame, preprocessor: StudentPreprocessor
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare data for prediction (no target column needed).

    Args:
        df: DataFrame with student data
        preprocessor: Fitted preprocessor

    Returns:
        Tuple of (feature array, DataFrame with derived features)
    """
    df_features = preprocessor.create_features(df)
    X = preprocessor.transform(df_features)
    return X, df_features
