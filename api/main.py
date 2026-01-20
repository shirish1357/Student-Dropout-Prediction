"""FastAPI application entry point."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.predictions import router as predictions_router
from api.routes.reports import router as reports_router
from api.services.predictor import get_predictor_service
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    predictor = get_predictor_service()
    loaded = predictor.load()
    if loaded:
        print(f"Model loaded successfully: {predictor.model_type}")
    else:
        print("Warning: Model not found. Train a model first:")
        print("  python main.py generate-sample")
        print("  python main.py train --data data/sample_students.csv")
    yield
app = FastAPI(
    title="Student Dropout Prediction API",
    description="API for predicting student dropout risk and identifying contributing factors.",
    version="1.0.0",
    lifespan=lifespan,
)
# Configure CORS - include Vercel domains
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:3000",
]
# Add Vercel URLs dynamically
vercel_url = os.environ.get("VERCEL_URL")
if vercel_url:
    allowed_origins.append(f"https://{vercel_url}")
    
# Allow all vercel.app subdomains in production
allowed_origins.append("https://*.vercel.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers
app.include_router(predictions_router)
app.include_router(reports_router)
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Student Dropout Prediction API",
        "docs": "/docs",
        "health": "/api/health",
        "predict": "/api/predict",
    }
