"""Configuration constants for the dropout prediction system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Data schema - required columns
REQUIRED_COLUMNS = [
    "student_id",
    "enrollment_date",
    "program",
    "enrollment_status",
    "gpa",
    "credits_attempted",
    "credits_completed",
    "failed_courses",
    "attendance_rate",
    "lms_logins_last_30d",
    "assignments_submitted",
    "assignments_total",
    "late_submissions",
    "advisor_meetings",
    "financial_aid",
]

# Target column (only present in training data)
TARGET_COLUMN = "dropped_out"

# Feature groups
NUMERIC_FEATURES = [
    "gpa",
    "credits_attempted",
    "credits_completed",
    "failed_courses",
    "attendance_rate",
    "lms_logins_last_30d",
    "assignments_submitted",
    "assignments_total",
    "late_submissions",
    "advisor_meetings",
]

CATEGORICAL_FEATURES = [
    "program",
    "enrollment_status",
]

BOOLEAN_FEATURES = [
    "financial_aid",
]

# Derived features (created during preprocessing)
DERIVED_FEATURES = [
    "completion_rate",      # credits_completed / credits_attempted
    "submission_rate",      # assignments_submitted / assignments_total
    "late_rate",           # late_submissions / assignments_submitted
    "days_enrolled",       # days since enrollment
]

# Model settings
RISK_PERCENTILE = 80  # Top 20% flagged as at-risk
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Sample data generation
SAMPLE_SIZE = 500
PROGRAMS = ["Computer Science", "Business", "Engineering", "Liberal Arts", "Nursing"]

# Email settings (for student outreach)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SES_SENDER_EMAIL = os.getenv("SES_SENDER_EMAIL", "support@university.edu")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "advising@university.edu")
SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "(555) 123-4567")
APPOINTMENT_LINK = os.getenv("APPOINTMENT_LINK", "https://university.edu/schedule")
