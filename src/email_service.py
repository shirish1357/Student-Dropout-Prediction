"""AWS SES email service for student outreach."""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from .config import AWS_REGION, SES_SENDER_EMAIL, OUTPUT_DIR
from .email_templates import format_template, list_templates


class EmailService:
    """AWS SES email service with dry-run support."""

    def __init__(
        self,
        aws_region: str = AWS_REGION,
        sender_email: str = SES_SENDER_EMAIL,
    ):
        """
        Initialize the email service.

        Args:
            aws_region: AWS region for SES
            sender_email: Verified sender email address
        """
        self.aws_region = aws_region
        self.sender_email = sender_email
        self.client = None
        self.email_log: List[Dict] = []

        if BOTO3_AVAILABLE:
            self.client = boto3.client("ses", region_name=aws_region)
        else:
            print("Warning: boto3 not installed. Only dry-run mode available.")

    def send_email(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: str,
        dry_run: bool = True,
    ) -> Tuple[bool, str]:
        """
        Send a single email via AWS SES.

        Args:
            to_email: Recipient email address
            subject: Email subject
            body_html: HTML body content
            body_text: Plain text body content
            dry_run: If True, log email without sending

        Returns:
            Tuple of (success: bool, message: str)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "to_email": to_email,
            "subject": subject,
            "dry_run": dry_run,
            "status": "pending",
            "message_id": None,
            "error": None,
        }

        if dry_run:
            log_entry["status"] = "dry_run"
            self.email_log.append(log_entry)
            return True, f"[DRY RUN] Would send to: {to_email}"

        if not BOTO3_AVAILABLE:
            log_entry["status"] = "error"
            log_entry["error"] = "boto3 not installed"
            self.email_log.append(log_entry)
            return False, "Error: boto3 not installed"

        if self.client is None:
            log_entry["status"] = "error"
            log_entry["error"] = "SES client not initialized"
            self.email_log.append(log_entry)
            return False, "Error: SES client not initialized"

        try:
            response = self.client.send_email(
                Source=self.sender_email,
                Destination={"ToAddresses": [to_email]},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {
                        "Text": {"Data": body_text, "Charset": "UTF-8"},
                        "Html": {"Data": body_html, "Charset": "UTF-8"},
                    },
                },
            )
            log_entry["status"] = "sent"
            log_entry["message_id"] = response.get("MessageId")
            self.email_log.append(log_entry)
            return True, f"Sent to {to_email} (ID: {response.get('MessageId')})"

        except ClientError as e:
            error_msg = e.response["Error"]["Message"]
            log_entry["status"] = "error"
            log_entry["error"] = error_msg
            self.email_log.append(log_entry)
            return False, f"Error sending to {to_email}: {error_msg}"

    def send_bulk_outreach(
        self,
        students_df: pd.DataFrame,
        template_name: str = "at_risk_outreach",
        dry_run: bool = True,
        rate_limit: float = 0.1,
    ) -> Dict[str, int]:
        """
        Send outreach emails to multiple students.

        Args:
            students_df: DataFrame with student info (must have email, first_name columns)
            template_name: Name of email template to use
            dry_run: If True, log emails without sending
            rate_limit: Seconds to wait between emails (SES limit: 14/sec)

        Returns:
            Dictionary with counts: {"sent": n, "failed": n, "skipped": n}
        """
        # Validate required columns
        required_cols = ["email", "first_name"]
        missing = set(required_cols) - set(students_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Validate template
        if template_name not in list_templates():
            raise ValueError(f"Unknown template: {template_name}")

        results = {"sent": 0, "failed": 0, "skipped": 0}

        print(f"\n{'=' * 50}")
        print(f"Email Outreach - {'DRY RUN' if dry_run else 'SENDING'}")
        print(f"Template: {template_name}")
        print(f"Recipients: {len(students_df)}")
        print(f"{'=' * 50}\n")

        for idx, row in students_df.iterrows():
            email = row.get("email")
            first_name = row.get("first_name", "Student")

            # Skip if no email
            if pd.isna(email) or not email:
                print(f"  [SKIP] No email for student")
                results["skipped"] += 1
                continue

            # Skip if opted out
            if row.get("opt_out", False):
                print(f"  [SKIP] {email} - opted out")
                results["skipped"] += 1
                continue

            # Determine risk level for template
            risk_score = row.get("risk_score", 0.5)
            if risk_score >= 0.8:
                risk_level = "High"
            elif risk_score >= 0.6:
                risk_level = "Moderate"
            else:
                risk_level = "Elevated"

            # Format email
            formatted = format_template(template_name, first_name, risk_level)

            # Send email
            success, message = self.send_email(
                to_email=email,
                subject=formatted["subject"],
                body_html=formatted["body_html"],
                body_text=formatted["body_text"],
                dry_run=dry_run,
            )

            if success:
                print(f"  [OK] {message}")
                results["sent"] += 1
            else:
                print(f"  [FAIL] {message}")
                results["failed"] += 1

            # Rate limiting (only when actually sending)
            if not dry_run and rate_limit > 0:
                time.sleep(rate_limit)

        # Summary
        print(f"\n{'=' * 50}")
        print(f"Results: {results['sent']} sent, {results['failed']} failed, {results['skipped']} skipped")
        print(f"{'=' * 50}\n")

        return results

    def save_log(self, filepath: Optional[str] = None) -> str:
        """
        Save email log to CSV file.

        Args:
            filepath: Path to save log (default: outputs/email_log.csv)

        Returns:
            Path to saved log file
        """
        if filepath is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            filepath = str(OUTPUT_DIR / "email_log.csv")

        if not self.email_log:
            print("No emails to log.")
            return filepath

        df = pd.DataFrame(self.email_log)
        df.to_csv(filepath, index=False)
        print(f"Email log saved to: {filepath}")
        return filepath

    def preview_email(
        self,
        student_name: str = "Student",
        template_name: str = "at_risk_outreach",
    ) -> None:
        """
        Preview an email template with sample data.

        Args:
            student_name: Name to use in preview
            template_name: Template to preview
        """
        formatted = format_template(template_name, student_name)

        print("\n" + "=" * 60)
        print("EMAIL PREVIEW")
        print("=" * 60)
        print(f"\nTemplate: {template_name}")
        print(f"From: {self.sender_email}")
        print(f"Subject: {formatted['subject']}")
        print("\n" + "-" * 40)
        print("PLAIN TEXT VERSION:")
        print("-" * 40)
        print(formatted["body_text"])
        print("\n" + "=" * 60 + "\n")
