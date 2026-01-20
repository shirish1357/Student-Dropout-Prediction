"""Pydantic models for student data validation."""

from datetime import date
from typing import List, Literal
from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    """Input schema for student dropout prediction."""

    student_id: str = Field(..., description="Unique student identifier")
    enrollment_date: date = Field(..., description="Date of enrollment (YYYY-MM-DD)")
    program: Literal[
        "Computer Science", "Business", "Engineering", "Liberal Arts", "Nursing"
    ] = Field(..., description="Degree program")
    enrollment_status: Literal["full_time", "part_time"] = Field(
        ..., description="Enrollment status"
    )
    gpa: float = Field(..., ge=0.0, le=4.0, description="Cumulative GPA (0.0-4.0)")
    credits_attempted: int = Field(..., ge=0, description="Total credits attempted")
    credits_completed: int = Field(..., ge=0, description="Total credits completed")
    failed_courses: int = Field(..., ge=0, description="Number of failed courses")
    attendance_rate: float = Field(
        ..., ge=0, le=100, description="Attendance percentage (0-100)"
    )
    lms_logins_last_30d: int = Field(
        ..., ge=0, description="LMS logins in last 30 days"
    )
    assignments_submitted: int = Field(
        ..., ge=0, description="Number of assignments submitted"
    )
    assignments_total: int = Field(
        ..., ge=1, description="Total assignments assigned"
    )
    late_submissions: int = Field(..., ge=0, description="Number of late submissions")
    advisor_meetings: int = Field(..., ge=0, description="Number of advisor meetings")
    financial_aid: bool = Field(..., description="Has financial aid")

    model_config = {
        "json_schema_extra": {
            "example": {
                "student_id": "STU00123",
                "enrollment_date": "2023-09-01",
                "program": "Computer Science",
                "enrollment_status": "full_time",
                "gpa": 2.8,
                "credits_attempted": 45,
                "credits_completed": 42,
                "failed_courses": 1,
                "attendance_rate": 78.5,
                "lms_logins_last_30d": 18,
                "assignments_submitted": 19,
                "assignments_total": 25,
                "late_submissions": 4,
                "advisor_meetings": 1,
                "financial_aid": True,
            }
        }
    }


class RiskFactor(BaseModel):
    """A single risk factor contributing to dropout prediction."""

    factor: str = Field(..., description="Name of the risk factor")
    severity: Literal["high", "medium", "low"] = Field(
        ..., description="Severity level"
    )
    value: str = Field(..., description="Current value and threshold info")


class PredictionResponse(BaseModel):
    """Response schema for dropout prediction."""

    student_id: str = Field(..., description="Student identifier")
    risk_score: float = Field(..., description="Dropout probability (0.0-1.0)")
    risk_percentage: int = Field(..., description="Risk as percentage (0-100)")
    risk_status: Literal["High Risk", "Medium Risk", "Low Risk"] = Field(
        ..., description="Risk classification"
    )
    risk_factors: List[RiskFactor] = Field(
        ..., description="Contributing risk factors"
    )
    recommendations: List[str] = Field(
        ..., description="Actionable recommendations"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "student_id": "STU00123",
                "risk_score": 0.742,
                "risk_percentage": 74,
                "risk_status": "High Risk",
                "risk_factors": [
                    {
                        "factor": "Low GPA",
                        "severity": "high",
                        "value": "1.8 (below 2.0 threshold)",
                    }
                ],
                "recommendations": [
                    "Schedule immediate meeting with academic advisor"
                ],
            }
        }
    }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str
    model_loaded: bool
    model_type: str | None = None
