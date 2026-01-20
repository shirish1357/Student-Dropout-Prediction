#!/usr/bin/env python3
"""
Student Dropout Prediction System

CLI tool for training models and predicting student dropout risk.

Usage:
    python main.py generate-sample              Generate sample data for testing
    python main.py train --data <path>          Train models on student data
    python main.py predict --data <path>        Predict dropout risk for students
    python main.py outreach --data <path>       Send outreach emails to at-risk students
"""

import argparse
import sys
from pathlib import Path

from src.config import (
    DATA_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    SAMPLE_SIZE,
    RISK_PERCENTILE,
)
from src.data_loader import load_student_data, save_sample_data
from src.preprocessor import (
    StudentPreprocessor,
    prepare_training_data,
    prepare_prediction_data,
)
from src.model import (
    train_models,
    evaluate_models,
    select_best_model,
    predict_risk_scores,
    identify_at_risk,
    get_feature_importance,
    save_model,
    load_model,
    print_evaluation_report,
)
from src.email_service import EmailService
from src.email_templates import list_templates


def cmd_generate_sample(args: argparse.Namespace) -> int:
    """Generate sample student data for testing."""
    output_path = DATA_DIR / "sample_students.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} sample student records...")
    save_sample_data(str(output_path), n=args.n)
    print(f"Sample data saved to: {output_path}")

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train dropout prediction models."""
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    print(f"Loading data from: {data_path}")
    df = load_student_data(str(data_path), has_target=True)
    print(f"Loaded {len(df)} student records")

    # Check dropout rate
    dropout_rate = df["dropped_out"].mean()
    print(f"Dropout rate in data: {dropout_rate:.1%}")

    # Prepare data
    print("\nPreparing training data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(df)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train models
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    print("Trained: Logistic Regression, Random Forest")

    # Evaluate
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)
    best_name, best_model = select_best_model(models, results)
    print_evaluation_report(results, best_name)

    # Feature importance
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    importance = get_feature_importance(
        best_model, preprocessor.get_feature_names(), best_name
    )
    for i, row in importance.head(10).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "dropout_model.joblib"
    save_model(best_model, preprocessor, str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Also save model type info
    model_info_path = MODEL_DIR / "model_info.txt"
    with open(model_info_path, "w") as f:
        f.write(f"model_type={best_name}\n")

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Predict dropout risk for students."""
    data_path = Path(args.data)
    model_path = MODEL_DIR / "dropout_model.joblib"
    model_info_path = MODEL_DIR / "model_info.txt"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    if not model_path.exists():
        print(f"Error: No trained model found. Run 'train' first.")
        return 1

    # Load model
    print(f"Loading model from: {model_path}")
    model, preprocessor = load_model(str(model_path))

    # Read model type
    model_type = "random_forest"  # default
    if model_info_path.exists():
        with open(model_info_path) as f:
            for line in f:
                if line.startswith("model_type="):
                    model_type = line.strip().split("=")[1]

    # Load data (may or may not have target column)
    print(f"Loading data from: {data_path}")
    try:
        df = load_student_data(str(data_path), has_target=True)
    except ValueError:
        df = load_student_data(str(data_path), has_target=False)

    print(f"Loaded {len(df)} student records")

    # Prepare data and predict
    print("\nGenerating risk predictions...")
    X, df_features = prepare_prediction_data(df, preprocessor)
    risk_scores = predict_risk_scores(model, X)

    # Identify at-risk students
    results = identify_at_risk(df, risk_scores, percentile=args.percentile)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "at_risk_students.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    at_risk_count = results["is_at_risk"].sum()
    threshold = (100 - args.percentile)
    print(f"\nSummary:")
    print(f"  Total students: {len(results)}")
    print(f"  At-risk (top {threshold}%): {at_risk_count}")
    print(f"  Risk score range: {risk_scores.min():.3f} - {risk_scores.max():.3f}")

    # Show top 10 at-risk
    print(f"\nTop 10 Highest Risk Students:")
    print("-" * 50)
    for i, row in results.head(10).iterrows():
        print(f"  {row['student_id']}: {row['risk_score']:.3f} (rank #{row['risk_rank']})")

    return 0


def cmd_outreach(args: argparse.Namespace) -> int:
    """Send outreach emails to at-risk students."""
    import pandas as pd

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    # Load data
    print(f"Loading student data from: {data_path}")
    df = pd.read_csv(data_path)

    # Check required columns
    required_cols = ["email", "first_name"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"Error: Missing required columns for outreach: {missing}")
        print("Required columns: email, first_name")
        print("Tip: Regenerate sample data or add these columns to your CSV")
        return 1

    # Filter by at-risk status or top N
    if args.all_at_risk:
        if "is_at_risk" not in df.columns:
            print("Error: 'is_at_risk' column not found. Run predict first.")
            return 1
        df = df[df["is_at_risk"] == True]
        print(f"Selected {len(df)} students flagged as at-risk")
    elif args.top:
        if "risk_score" not in df.columns:
            print("Error: 'risk_score' column not found. Run predict first.")
            return 1
        df = df.nlargest(args.top, "risk_score")
        print(f"Selected top {len(df)} highest-risk students")

    if len(df) == 0:
        print("No students to contact.")
        return 0

    # Preview mode
    if args.preview:
        email_service = EmailService()
        email_service.preview_email(
            student_name="Sample Student",
            template_name=args.template,
        )
        return 0

    # Confirmation for actual sending
    if not args.dry_run:
        print(f"\nYou are about to send {len(df)} emails.")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return 0

    # Send emails
    email_service = EmailService()
    results = email_service.send_bulk_outreach(
        students_df=df,
        template_name=args.template,
        dry_run=args.dry_run,
    )

    # Save log
    email_service.save_log()

    return 0 if results["failed"] == 0 else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Student Dropout Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate-sample command
    gen_parser = subparsers.add_parser(
        "generate-sample", help="Generate sample student data"
    )
    gen_parser.add_argument(
        "-n", type=int, default=SAMPLE_SIZE, help=f"Number of students (default: {SAMPLE_SIZE})"
    )

    # train command
    train_parser = subparsers.add_parser("train", help="Train dropout prediction models")
    train_parser.add_argument(
        "--data", "-d", required=True, help="Path to training data CSV"
    )

    # predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict dropout risk for students"
    )
    predict_parser.add_argument(
        "--data", "-d", required=True, help="Path to student data CSV"
    )
    predict_parser.add_argument(
        "--percentile",
        "-p",
        type=int,
        default=RISK_PERCENTILE,
        help=f"Risk percentile threshold (default: {RISK_PERCENTILE})",
    )

    # outreach command
    outreach_parser = subparsers.add_parser(
        "outreach", help="Send outreach emails to at-risk students"
    )
    outreach_parser.add_argument(
        "--data", "-d", required=True, help="Path to student data CSV (with risk scores)"
    )
    outreach_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview emails without sending (default: True)",
    )
    outreach_parser.add_argument(
        "--send",
        action="store_false",
        dest="dry_run",
        help="Actually send emails (disables dry-run)",
    )
    outreach_parser.add_argument(
        "--top",
        type=int,
        help="Send to top N highest-risk students",
    )
    outreach_parser.add_argument(
        "--all-at-risk",
        action="store_true",
        help="Send to all students flagged as at-risk",
    )
    outreach_parser.add_argument(
        "--template",
        default="at_risk_outreach",
        choices=list_templates(),
        help="Email template to use (default: at_risk_outreach)",
    )
    outreach_parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the email template without sending",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "generate-sample":
        return cmd_generate_sample(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "predict":
        return cmd_predict(args)
    elif args.command == "outreach":
        return cmd_outreach(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
