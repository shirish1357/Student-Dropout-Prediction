"""Report generation API routes."""

import io
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse

from api.schemas.student import StudentInput, PredictionResponse
from api.services.predictor import PredictorService, get_predictor_service
from api.services.pdf_generator import PDFGeneratorService, get_pdf_generator

router = APIRouter(prefix="/api", tags=["reports"])


@router.post("/report/single")
async def generate_single_report(
    student: StudentInput,
    predictor: PredictorService = Depends(get_predictor_service),
    pdf_generator: PDFGeneratorService = Depends(get_pdf_generator),
) -> StreamingResponse:
    """
    Generate a PDF report for a single student.

    Takes student data, generates prediction, and returns a detailed PDF report
    with charts including risk gauge, risk factors bar chart, and radar chart.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first.",
        )

    try:
        # Generate prediction
        prediction = predictor.predict(student)

        # Generate PDF
        pdf_bytes = pdf_generator.generate_single_report(prediction, student)

        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=risk_report_{student.student_id}.pdf"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}",
        )


@router.post("/report/batch")
async def generate_batch_report(
    file: UploadFile = File(...),
    predictor: PredictorService = Depends(get_predictor_service),
    pdf_generator: PDFGeneratorService = Depends(get_pdf_generator),
) -> StreamingResponse:
    """
    Generate a PDF summary report for batch predictions.

    Accepts a CSV file with student data and returns a PDF report with:
    - Risk distribution pie chart
    - Risk by program bar chart
    - High-risk students table
    - Common risk factors analysis
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

        # Process each student
        predictions: List[PredictionResponse] = []
        students: List[StudentInput] = []

        for _, row in df.iterrows():
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

            students.append(student)
            prediction = predictor.predict(student)
            predictions.append(prediction)

        # Generate PDF
        pdf_bytes = pdf_generator.generate_batch_report(predictions, students)

        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=batch_risk_report.pdf"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch report generation failed: {str(e)}",
        )
