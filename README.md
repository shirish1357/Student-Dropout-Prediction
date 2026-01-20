# Student Dropout Prediction System

A machine learning system to identify students at risk of dropping out, enabling proactive outreach and intervention.

## Overview

This system analyzes student enrollment, academic performance, and engagement data to predict dropout risk. Students flagged as high-risk (top 20%) can be prioritized for personalized support and intervention.

**Key Features:**
- CSV data ingestion with validation
- Feature engineering (completion rates, engagement metrics)
- Multi-model training (Logistic Regression, Random Forest, XGBoost)
- Automated hyperparameter tuning with cross-validation
- SMOTE oversampling for class imbalance handling
- Automatic model selection based on ROC-AUC score
- Risk score generation and at-risk student identification
- Web application for interactive predictions (FastAPI + React)
- Interactive Jupyter notebook for EDA and model exploration

## Installation

```bash
# Clone or navigate to the project directory
cd student-dropout-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate sample data for testing
python main.py generate-sample

# 2. Train the model
python main.py train --data data/sample_students.csv

# 3. Generate predictions
python main.py predict --data data/sample_students.csv

# 4. View results
cat outputs/at_risk_students.csv
```

## Usage

### Generate Sample Data
```bash
python main.py generate-sample -n 500
```
Creates synthetic student data at `data/sample_students.csv` for testing.

### Train Models
```bash
python main.py train --data path/to/your/students.csv
```
Trains both Logistic Regression and Random Forest models, selects the best one, and saves it to `models/dropout_model.joblib`.

### Generate Predictions
```bash
python main.py predict --data path/to/students.csv --percentile 80
```
Outputs `outputs/at_risk_students.csv` with risk scores and at-risk flags.

### Jupyter Notebook
```bash
jupyter notebook notebooks/eda_and_modeling.ipynb
```
Interactive walkthrough of EDA, feature engineering, and model building.

## Data Format

Your CSV file should contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| student_id | string | Unique identifier |
| enrollment_date | date | When student enrolled (YYYY-MM-DD) |
| program | string | Degree program name |
| enrollment_status | string | "full_time" or "part_time" |
| gpa | float | Cumulative GPA (0.0-4.0) |
| credits_attempted | int | Total credits attempted |
| credits_completed | int | Total credits completed |
| failed_courses | int | Number of failed courses |
| attendance_rate | float | Attendance percentage (0-100) |
| lms_logins_last_30d | int | LMS logins in last 30 days |
| assignments_submitted | int | Number of assignments submitted |
| assignments_total | int | Total assignments assigned |
| late_submissions | int | Number of late submissions |
| advisor_meetings | int | Number of advisor meetings |
| financial_aid | bool | Has financial aid (true/false) |
| dropped_out | bool | Target variable (for training only) |

## Output Format

The `at_risk_students.csv` output contains:

| Column | Description |
|--------|-------------|
| student_id | Student identifier |
| risk_score | Probability of dropout (0-1) |
| risk_rank | Rank by risk (1 = highest risk) |
| is_at_risk | True if in top 20% risk |

## Model Interpretation

### Key Risk Factors (typical findings)

**Increases dropout risk:**
- Low GPA (< 2.0)
- Low attendance rate (< 70%)
- Few LMS logins
- Low assignment submission rate
- Multiple failed courses

**Decreases dropout risk:**
- High GPA (> 3.0)
- Regular LMS engagement
- High assignment completion
- Advisor meeting attendance
- Financial aid (provides stability)

### Using Feature Importance

After training, the system prints the top 10 most important features. Use this to:
1. Understand which factors drive predictions
2. Guide intervention strategies (e.g., if attendance is key, focus on attendance tracking)
3. Identify data quality issues (unexpected importance may indicate data problems)

## Hyperparameter Tuning

The system performs automated hyperparameter tuning using 5-fold stratified cross-validation with ROC-AUC as the optimization metric.

### Models and Hyperparameters

**Logistic Regression:**
| Parameter | Values Tested |
|-----------|---------------|
| C (regularization) | 0.01, 0.1, 1.0, 10.0 |
| Penalty | L1, L2 |
| Solver | lbfgs (L2), saga (L1) |

**Random Forest:**
| Parameter | Values Tested |
|-----------|---------------|
| n_estimators | 100, 200 |
| max_depth | 5, 10, 15 |
| min_samples_split | 2, 5, 10 |

**XGBoost:**
| Parameter | Values Tested |
|-----------|---------------|
| n_estimators | 100, 200, 300 |
| max_depth | 3, 5, 7 |
| learning_rate | 0.03, 0.05, 0.1 |
| subsample | 0.8, 0.9 |

### Class Imbalance Handling

The system uses **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance in training data. This generates synthetic examples of the minority class (dropout students) to create a balanced training set, improving model performance on underrepresented cases.

### Model Selection

After tuning, all models are evaluated on a held-out test set (20% of data). The model with the highest **ROC-AUC score** is automatically selected and saved. ROC-AUC is preferred over accuracy because it better captures performance on imbalanced datasets.

## Project Structure

```
student-dropout-prediction/
├── api/                          # FastAPI backend
│   ├── main.py                   # API entry point
│   ├── routes/
│   │   └── predictions.py        # Prediction endpoints
│   ├── services/
│   │   └── predictor.py          # Model loading & prediction
│   └── schemas/
│       └── student.py            # Pydantic validation models
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── App.jsx               # Main app component
│   │   ├── components/           # UI components
│   │   └── services/api.js       # API client
│   └── package.json
├── data/
│   ├── raw/                      # Original CSV files
│   ├── processed/                # Cleaned data
│   └── sample_students.csv       # Generated sample data
├── notebooks/
│   └── eda_and_modeling.ipynb    # EDA and model walkthrough
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration constants
│   ├── data_loader.py            # Data loading and validation
│   ├── preprocessor.py           # Feature engineering
│   └── model.py                  # Model training and evaluation
├── models/
│   └── dropout_model.joblib      # Saved trained model
├── outputs/
│   └── at_risk_students.csv      # Prediction results
├── main.py                       # CLI entry point
├── requirements.txt
├── RUN_APP.md                    # Web app run instructions
└── README.md
```

## Configuration

Edit `src/config.py` to customize:
- `RISK_PERCENTILE`: Threshold for at-risk flagging (default: 80 = top 20%)
- `TEST_SIZE`: Train/test split ratio (default: 0.2)
- `SAMPLE_SIZE`: Default sample data size (default: 500)

## Web Application

The system includes a web interface for interactive predictions.

### Running the Web App

**Terminal 1 - Start Backend:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
cd frontend && npm run dev
```

Open **http://localhost:5173** in your browser.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Submit student data, get risk prediction |
| `/api/health` | GET | Check API and model status |
| `/docs` | GET | Interactive Swagger documentation |

### Prediction Response

The API returns detailed predictions including:
- **Risk score** (0-100%)
- **Risk status** (High/Medium/Low)
- **Risk factors** with severity levels
- **Personalized recommendations**

## Next Steps

1. **Integrate with student information system**: Replace CSV ingestion with database connection
2. **Email integration**: Connect to email service for automated outreach
3. **Feedback loop**: Track intervention outcomes to improve model over time
4. **Authentication**: Add user login for the web application

## Troubleshooting

**"Missing required columns" error:**
- Ensure your CSV has all required columns (see Data Format section)
- Column names are case-sensitive

**Low model performance:**
- Check class imbalance in your data
- Ensure data quality (no missing values, reasonable ranges)
- Consider adding more relevant features

**Memory issues with large datasets:**
- Process data in batches
- Use `generate-sample` with smaller `-n` for testing
