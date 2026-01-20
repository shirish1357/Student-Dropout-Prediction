# Running the Student Dropout Prediction App

## Prerequisites

Make sure dependencies are installed:

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install
```

## Start the Application

### Terminal 1 - Start Backend (FastAPI)

```bash
cd "/Users/shirishghimire/projects/claude test/student-dropout-prediction"
uvicorn api.main:app --reload --port 8000
```

### Terminal 2 - Start Frontend (React)

```bash
cd "/Users/shirishghimire/projects/claude test/student-dropout-prediction/frontend"
npm run dev
```

## Access the App

Open your browser and go to: **http://localhost:5173**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check API and model status |
| `/api/predict` | POST | Submit student data for prediction |
| `/docs` | GET | Interactive API documentation (Swagger) |
