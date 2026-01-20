"""Data loading and validation utilities."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from .config import (
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    PROGRAMS,
    SAMPLE_SIZE,
    RANDOM_STATE,
)


def load_student_data(filepath: str, has_target: bool = True) -> pd.DataFrame:
    """
    Load student data from a CSV file.

    Args:
        filepath: Path to the CSV file
        has_target: Whether the data includes the dropout target column

    Returns:
        DataFrame with student data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing
    """
    df = pd.read_csv(filepath)
    validate_schema(df, has_target=has_target)

    # Parse date column
    df["enrollment_date"] = pd.to_datetime(df["enrollment_date"])

    # Convert boolean column
    df["financial_aid"] = df["financial_aid"].astype(bool)

    if has_target:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(bool)

    return df


def validate_schema(df: pd.DataFrame, has_target: bool = True) -> None:
    """
    Validate that the DataFrame has all required columns.

    Args:
        df: DataFrame to validate
        has_target: Whether to check for target column

    Raises:
        ValueError: If required columns are missing
    """
    required = REQUIRED_COLUMNS.copy()
    if has_target:
        required.append(TARGET_COLUMN)

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def generate_sample_data(
    n: int = SAMPLE_SIZE,
    dropout_rate: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate synthetic student data for testing.

    Creates STRONG, realistic correlations between features and dropout using
    a latent risk factor approach with reduced noise. Dropout is determined
    by a logistic function of multiple weighted features.

    Args:
        n: Number of students to generate
        dropout_rate: Target overall dropout rate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with synthetic student data
    """
    np.random.seed(random_state)

    # Sample first names for realistic data
    first_names = [
        "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
        "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
        "Lucas", "Harper", "Henry", "Evelyn", "Alexander", "Abigail", "Michael",
        "Emily", "Daniel", "Elizabeth", "Jacob", "Sofia", "Logan", "Avery", "Jackson",
        "Ella", "Sebastian", "Madison", "Aiden", "Scarlett", "Matthew", "Victoria",
        "Samuel", "Aria", "David", "Grace", "Joseph", "Chloe", "Carter", "Camila",
        "Owen", "Penelope", "Wyatt", "Riley", "John", "Layla", "Jack", "Lillian",
    ]

    # Generate base features
    data = {
        "student_id": [f"STU{str(i).zfill(5)}" for i in range(1, n + 1)],
        "first_name": np.random.choice(first_names, n),
        "email": [f"student{str(i).zfill(5)}@university.edu" for i in range(1, n + 1)],
        "enrollment_date": [
            datetime.now() - timedelta(days=np.random.randint(30, 730))
            for _ in range(n)
        ],
    }

    # Latent risk factor - this drives most features
    # Use mixture of two populations: low-risk majority and high-risk minority
    is_high_risk = np.random.binomial(1, 0.25, n).astype(bool)
    risk_factor = np.where(
        is_high_risk,
        np.random.beta(5, 2, n),  # High risk group: skewed high
        np.random.beta(2, 8, n),  # Low risk group: skewed low
    )

    # Program affects dropout (some programs harder)
    program_risk = {
        "Computer Science": 0.0,
        "Business": -0.05,
        "Engineering": 0.1,
        "Liberal Arts": -0.03,
        "Nursing": 0.05,
    }
    data["program"] = np.random.choice(PROGRAMS, n)
    program_risk_values = np.array([program_risk[p] for p in data["program"]])

    # Enrollment status (part-time has higher dropout)
    is_part_time = np.random.binomial(1, 0.25, n).astype(bool)
    data["enrollment_status"] = np.where(is_part_time, "part_time", "full_time")
    enrollment_risk = np.where(is_part_time, 0.15, 0.0)

    # Financial aid (protective factor)
    # High-risk students less likely to have aid
    aid_prob = np.where(risk_factor > 0.5, 0.4, 0.7)
    data["financial_aid"] = np.random.binomial(1, aid_prob).astype(bool)
    aid_protection = np.where(data["financial_aid"], -0.1, 0.0)

    # GPA: STRONG inverse correlation with risk (low noise)
    noise_gpa = np.random.normal(0, 0.15, n)
    data["gpa"] = np.clip(3.8 - (risk_factor * 2.2) + noise_gpa, 0.5, 4.0)

    # Credits attempted (varies by enrollment length)
    days_enrolled = np.array([(datetime.now() - d).days for d in
                               [datetime.now() - timedelta(days=np.random.randint(30, 730)) for _ in range(n)]])
    data["credits_attempted"] = np.clip(
        (days_enrolled / 365 * 30 + np.random.normal(30, 10, n)).astype(int),
        15, 120
    )

    # Credits completed: tightly correlated with GPA and risk
    completion_rate = np.clip(0.98 - (risk_factor * 0.4) + np.random.normal(0, 0.03, n), 0.4, 1.0)
    data["credits_completed"] = (data["credits_attempted"] * completion_rate).astype(int)

    # Failed courses: strongly correlated with risk
    data["failed_courses"] = np.clip(
        np.round(risk_factor * 6 + np.random.normal(0, 0.3, n)), 0, 8
    ).astype(int)

    # Attendance rate: strong inverse correlation
    data["attendance_rate"] = np.clip(
        98 - (risk_factor * 45) + np.random.normal(0, 3, n), 30, 100
    )

    # LMS engagement: strong inverse correlation with threshold effect
    base_logins = 35 - (risk_factor * 30)
    # Add "disengaged" effect - some high-risk students stop logging in entirely
    disengaged = (risk_factor > 0.6) & (np.random.random(n) < 0.4)
    base_logins = np.where(disengaged, base_logins * 0.2, base_logins)
    data["lms_logins_last_30d"] = np.clip(
        base_logins + np.random.normal(0, 2, n), 0, 50
    ).astype(int)

    # Assignment metrics with strong correlation
    data["assignments_total"] = np.random.randint(20, 35, n)
    submission_rate = np.clip(0.98 - (risk_factor * 0.5) + np.random.normal(0, 0.03, n), 0.2, 1.0)
    data["assignments_submitted"] = (data["assignments_total"] * submission_rate).astype(int)

    # Late submissions: strongly correlated with risk
    max_late = data["assignments_submitted"]
    late_rate = np.clip(risk_factor * 0.6 + np.random.normal(0, 0.05, n), 0, 0.9)
    data["late_submissions"] = (max_late * late_rate).astype(int)

    # Advisor meetings: inverse correlation (struggling students avoid help)
    data["advisor_meetings"] = np.clip(
        6 - (risk_factor * 5) + np.random.normal(0, 0.5, n), 0, 8
    ).astype(int)

    # Calculate dropout using logistic function of multiple features
    # This creates learnable patterns for the model
    gpa_normalized = (data["gpa"] - 2.5) / 1.5
    attendance_normalized = (data["attendance_rate"] - 75) / 25
    lms_normalized = (data["lms_logins_last_30d"] - 20) / 15
    submission_normalized = (data["assignments_submitted"] / data["assignments_total"] - 0.7) / 0.3

    # Weighted combination (these weights make the relationship learnable)
    logit = (
        -1.5  # Base (controls overall dropout rate)
        - 1.8 * gpa_normalized  # GPA is strong predictor
        - 1.2 * attendance_normalized  # Attendance matters
        - 0.8 * lms_normalized  # Engagement helps
        - 0.6 * submission_normalized  # Submissions matter
        + 0.4 * data["failed_courses"]  # Failed courses hurt
        + enrollment_risk * 3  # Part-time risk
        + program_risk_values * 2  # Program effects
        + aid_protection * 2  # Financial aid helps
        + np.random.normal(0, 0.3, n)  # Small noise
    )

    # Convert to probability via sigmoid
    dropout_prob = 1 / (1 + np.exp(-logit))

    # Generate outcomes
    data["dropped_out"] = np.random.binomial(1, dropout_prob).astype(bool)

    df = pd.DataFrame(data)

    # Round float columns
    df["gpa"] = df["gpa"].round(2)
    df["attendance_rate"] = df["attendance_rate"].round(1)

    # Reorder columns
    column_order = [
        "student_id", "first_name", "email", "enrollment_date", "program",
        "enrollment_status", "gpa", "credits_attempted", "credits_completed",
        "failed_courses", "attendance_rate", "lms_logins_last_30d",
        "assignments_submitted", "assignments_total", "late_submissions",
        "advisor_meetings", "financial_aid", "dropped_out"
    ]
    df = df[column_order]

    return df


def save_sample_data(filepath: str, n: int = SAMPLE_SIZE) -> str:
    """
    Generate and save sample data to a CSV file.

    Args:
        filepath: Where to save the CSV
        n: Number of students to generate

    Returns:
        Path to the saved file
    """
    df = generate_sample_data(n=n)
    df.to_csv(filepath, index=False)
    return filepath
