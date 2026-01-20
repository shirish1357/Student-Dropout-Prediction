"""Prediction API routes."""

import io
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel

from api.schemas.student import StudentInput, PredictionResponse, HealthResponse
from api.services.predictor import PredictorService, get_predictor_service

router = APIRouter(prefix="/api", tags=["predictions"])


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    total: int
    high_risk: int
    medium_risk: int
    low_risk: int
    predictions: List[PredictionResponse]


@router.post("/predict", response_model=PredictionResponse)
async def predict_dropout(
    student: StudentInput,
    predictor: PredictorService = Depends(get_predictor_service),
) -> PredictionResponse:
    """
    Predict dropout risk for a student.

    Takes student data as input and returns:
    - Risk score (0-1 probability)
    - Risk status (High/Medium/Low)
    - Contributing risk factors with severity
    - Personalized recommendations
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first using: python main.py train --data data/sample_students.csv",
        )

    try:
        return predictor.predict(student)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    predictor: PredictorService = Depends(get_predictor_service),
) -> BatchPredictionResponse:
    """
    Predict dropout risk for multiple students from a CSV file.

    Accepts a CSV file with student data and returns predictions for all students.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first.",
        )

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Required columns
        required_cols = [
            'student_id', 'enrollment_date', 'program', 'enrollment_status',
            'gpa', 'credits_attempted', 'credits_completed', 'failed_courses',
            'attendance_rate', 'lms_logins_last_30d', 'assignments_submitted',
            'assignments_total', 'late_submissions', 'advisor_meetings', 'financial_aid'
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_cols)}",
            )

        predictions = []
        high_risk = 0
        medium_risk = 0
        low_risk = 0

        for _, row in df.iterrows():
            # Convert row to StudentInput
            student = StudentInput(
                student_id=str(row['student_id']),
                enrollment_date=pd.to_datetime(row['enrollment_date']).date(),
                program=row['program'],
                enrollment_status=row['enrollment_status'],
                gpa=float(row['gpa']),
                credits_attempted=int(row['credits_attempted']),
                credits_completed=int(row['credits_completed']),
                failed_courses=int(row['failed_courses']),
                attendance_rate=float(row['attendance_rate']),
                lms_logins_last_30d=int(row['lms_logins_last_30d']),
                assignments_submitted=int(row['assignments_submitted']),
                assignments_total=int(row['assignments_total']),
                late_submissions=int(row['late_submissions']),
                advisor_meetings=int(row['advisor_meetings']),
                financial_aid=bool(row['financial_aid']),
            )

            prediction = predictor.predict(student)
            predictions.append(prediction)

            if prediction.risk_status == "High Risk":
                high_risk += 1
            elif prediction.risk_status == "Medium Risk":
                medium_risk += 1
            else:
                low_risk += 1

        # Sort by risk score descending
        predictions.sort(key=lambda x: x.risk_score, reverse=True)

        return BatchPredictionResponse(
            total=len(predictions),
            high_risk=high_risk,
            medium_risk=medium_risk,
            low_risk=low_risk,
            predictions=predictions,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    predictor: PredictorService = Depends(get_predictor_service),
) -> HealthResponse:
    """
    Check API health and model status.

    Returns the current status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
        model_type=predictor.model_type if predictor.is_loaded else None,
    )
