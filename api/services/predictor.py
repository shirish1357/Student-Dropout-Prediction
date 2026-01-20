"""Prediction service for student dropout risk."""

import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_model
from src.preprocessor import prepare_prediction_data
from api.schemas.student import StudentInput, RiskFactor, PredictionResponse


class PredictorService:
    """Service for loading model and making predictions."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_type: str | None = None
        self._loaded = False

    def load(self, model_path: str | None = None) -> bool:
        """Load the trained model and preprocessor."""
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "dropout_model.joblib")

        try:
            self.model, self.preprocessor = load_model(model_path)
            self._loaded = True

            # Try to read model type from info file
            model_info_path = PROJECT_ROOT / "models" / "model_info.txt"
            if model_info_path.exists():
                content = model_info_path.read_text().strip()
                # Parse "model_type=xxx" format
                if "=" in content:
                    self.model_type = content.split("=", 1)[1]
                else:
                    self.model_type = content
            else:
                self.model_type = type(self.model).__name__

            return True
        except FileNotFoundError:
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def predict(self, student: StudentInput) -> PredictionResponse:
        """Generate dropout prediction for a single student."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert input to DataFrame
        df = self._input_to_dataframe(student)

        # Preprocess and predict
        X, _ = prepare_prediction_data(df, self.preprocessor)
        risk_score = float(self.model.predict_proba(X)[0, 1])

        # Analyze risk factors
        risk_factors = self._analyze_risk_factors(student)

        # Generate recommendations
        recommendations = self._generate_recommendations(risk_factors)

        # Determine risk status
        risk_percentage = int(risk_score * 100)
        if risk_score >= 0.7:
            risk_status = "High Risk"
        elif risk_score >= 0.4:
            risk_status = "Medium Risk"
        else:
            risk_status = "Low Risk"

        return PredictionResponse(
            student_id=student.student_id,
            risk_score=round(risk_score, 3),
            risk_percentage=risk_percentage,
            risk_status=risk_status,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )

    def _input_to_dataframe(self, student: StudentInput) -> pd.DataFrame:
        """Convert StudentInput to pandas DataFrame."""
        data = {
            "student_id": [student.student_id],
            "enrollment_date": [student.enrollment_date],
            "program": [student.program],
            "enrollment_status": [student.enrollment_status],
            "gpa": [student.gpa],
            "credits_attempted": [student.credits_attempted],
            "credits_completed": [student.credits_completed],
            "failed_courses": [student.failed_courses],
            "attendance_rate": [student.attendance_rate],
            "lms_logins_last_30d": [student.lms_logins_last_30d],
            "assignments_submitted": [student.assignments_submitted],
            "assignments_total": [student.assignments_total],
            "late_submissions": [student.late_submissions],
            "advisor_meetings": [student.advisor_meetings],
            "financial_aid": [student.financial_aid],
        }
        return pd.DataFrame(data)

    def _analyze_risk_factors(self, student: StudentInput) -> List[RiskFactor]:
        """Analyze input values to identify risk factors."""
        factors = []

        # GPA analysis
        if student.gpa < 2.0:
            factors.append(
                RiskFactor(
                    factor="Low GPA",
                    severity="high",
                    value=f"{student.gpa:.2f} (below 2.0 threshold)",
                )
            )
        elif student.gpa < 2.5:
            factors.append(
                RiskFactor(
                    factor="Below Average GPA",
                    severity="medium",
                    value=f"{student.gpa:.2f} (below 2.5 threshold)",
                )
            )

        # Attendance analysis
        if student.attendance_rate < 70:
            factors.append(
                RiskFactor(
                    factor="Low Attendance",
                    severity="high",
                    value=f"{student.attendance_rate:.1f}% (below 70% threshold)",
                )
            )
        elif student.attendance_rate < 80:
            factors.append(
                RiskFactor(
                    factor="Below Average Attendance",
                    severity="medium",
                    value=f"{student.attendance_rate:.1f}% (below 80% threshold)",
                )
            )

        # LMS engagement analysis
        if student.lms_logins_last_30d < 5:
            factors.append(
                RiskFactor(
                    factor="Very Low LMS Engagement",
                    severity="high",
                    value=f"{student.lms_logins_last_30d} logins (below 5 threshold)",
                )
            )
        elif student.lms_logins_last_30d < 15:
            factors.append(
                RiskFactor(
                    factor="Low LMS Engagement",
                    severity="medium",
                    value=f"{student.lms_logins_last_30d} logins (below 15 threshold)",
                )
            )

        # Assignment submission rate
        submission_rate = student.assignments_submitted / student.assignments_total
        if submission_rate < 0.6:
            factors.append(
                RiskFactor(
                    factor="Low Assignment Completion",
                    severity="high",
                    value=f"{submission_rate:.0%} submitted (below 60% threshold)",
                )
            )
        elif submission_rate < 0.8:
            factors.append(
                RiskFactor(
                    factor="Below Average Assignment Completion",
                    severity="medium",
                    value=f"{submission_rate:.0%} submitted (below 80% threshold)",
                )
            )

        # Failed courses
        if student.failed_courses > 2:
            factors.append(
                RiskFactor(
                    factor="Multiple Failed Courses",
                    severity="high",
                    value=f"{student.failed_courses} courses failed",
                )
            )
        elif student.failed_courses >= 1:
            factors.append(
                RiskFactor(
                    factor="Failed Course(s)",
                    severity="medium",
                    value=f"{student.failed_courses} course(s) failed",
                )
            )

        # Late submissions
        if student.assignments_submitted > 0:
            late_rate = student.late_submissions / student.assignments_submitted
            if late_rate > 0.3:
                factors.append(
                    RiskFactor(
                        factor="High Late Submission Rate",
                        severity="high",
                        value=f"{late_rate:.0%} of submissions late",
                    )
                )
            elif late_rate > 0.15:
                factors.append(
                    RiskFactor(
                        factor="Elevated Late Submission Rate",
                        severity="medium",
                        value=f"{late_rate:.0%} of submissions late",
                    )
                )

        # Advisor meetings
        if student.advisor_meetings == 0:
            factors.append(
                RiskFactor(
                    factor="No Advisor Meetings",
                    severity="medium",
                    value="0 meetings this term",
                )
            )

        # Financial aid
        if not student.financial_aid:
            factors.append(
                RiskFactor(
                    factor="No Financial Aid",
                    severity="low",
                    value="May face financial barriers",
                )
            )

        # Credit completion rate
        if student.credits_attempted > 0:
            completion_rate = student.credits_completed / student.credits_attempted
            if completion_rate < 0.7:
                factors.append(
                    RiskFactor(
                        factor="Low Credit Completion",
                        severity="high",
                        value=f"{completion_rate:.0%} of attempted credits completed",
                    )
                )
            elif completion_rate < 0.9:
                factors.append(
                    RiskFactor(
                        factor="Below Average Credit Completion",
                        severity="medium",
                        value=f"{completion_rate:.0%} of attempted credits completed",
                    )
                )

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        factors.sort(key=lambda x: severity_order[x.severity])

        return factors

    def _generate_recommendations(self, risk_factors: List[RiskFactor]) -> List[str]:
        """Generate personalized recommendations based on risk factors."""
        recommendations = []
        factor_names = {f.factor for f in risk_factors}

        # GPA-related recommendations
        if "Low GPA" in factor_names or "Below Average GPA" in factor_names:
            recommendations.append(
                "Enroll in tutoring services for struggling courses"
            )
            recommendations.append(
                "Consider reducing course load next semester"
            )

        # Attendance recommendations
        if "Low Attendance" in factor_names or "Below Average Attendance" in factor_names:
            recommendations.append(
                "Set up attendance tracking reminders"
            )
            recommendations.append(
                "Discuss attendance barriers with academic advisor"
            )

        # Engagement recommendations
        if "Very Low LMS Engagement" in factor_names or "Low LMS Engagement" in factor_names:
            recommendations.append(
                "Set up daily LMS login reminders"
            )
            recommendations.append(
                "Review online course materials weekly"
            )

        # Assignment recommendations
        if "Low Assignment Completion" in factor_names or "Below Average Assignment Completion" in factor_names:
            recommendations.append(
                "Create assignment tracking system with due dates"
            )
            recommendations.append(
                "Break large assignments into smaller tasks"
            )

        # Late submission recommendations
        if "High Late Submission Rate" in factor_names or "Elevated Late Submission Rate" in factor_names:
            recommendations.append(
                "Set personal deadlines 2 days before actual due dates"
            )

        # Advisor meeting recommendations
        if "No Advisor Meetings" in factor_names:
            recommendations.append(
                "Schedule meeting with academic advisor this week"
            )

        # Financial recommendations
        if "No Financial Aid" in factor_names:
            recommendations.append(
                "Explore financial aid options with student services"
            )

        # Failed courses recommendations
        if "Multiple Failed Courses" in factor_names or "Failed Course(s)" in factor_names:
            recommendations.append(
                "Meet with professor(s) during office hours for additional help"
            )

        # Always recommend advisor meeting for high-risk students
        if any(f.severity == "high" for f in risk_factors):
            if "Schedule meeting with academic advisor this week" not in recommendations:
                recommendations.insert(
                    0, "Schedule immediate meeting with academic advisor"
                )

        # Limit to top 5 recommendations
        return recommendations[:5]


# Singleton instance
_predictor_service: PredictorService | None = None


def get_predictor_service() -> PredictorService:
    """Get the singleton predictor service instance."""
    global _predictor_service
    if _predictor_service is None:
        _predictor_service = PredictorService()
    return _predictor_service
