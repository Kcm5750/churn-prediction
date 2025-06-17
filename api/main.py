from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import pandas as pd
import numpy as np
import joblib
import os
import sys
import uuid
import logging
from datetime import datetime

# Add the parent directory to the path for importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from src
from src.predict import ChurnPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_logs.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using a machine learning model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize paths

MODEL_PATH = os.environ.get("MODEL_PATH", "/home/kcm5750/Music/ChurnPrediction/api/model.joblib")
PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "/home/kcm5750/Music/ChurnPrediction/data/processed/preprocessing_pipeline.joblib")
BATCH_RESULTS_DIR = os.environ.get("BATCH_RESULTS_DIR", "batch_results")
os.makedirs(BATCH_RESULTS_DIR, exist_ok=True)
# Pydantic models for request and response
class CustomerData(BaseModel):
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[int] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None
    customerID: Optional[str] = None

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    explanation: Optional[List[Dict[str, Union[str, float]]]] = None
    request_id: str

class BatchPredictionResponse(BaseModel):
    job_id: str
    status: str
    message: str

class BatchPredictionStatus(BaseModel):
    job_id: str
    status: str
    total_records: Optional[int] = None
    completed_records: Optional[int] = None
    result_file: Optional[str] = None

# Global job status tracker
batch_jobs = {}

# Function to get predictor instance
def get_predictor():
    try:
        predictor = ChurnPredictor(MODEL_PATH, PIPELINE_PATH)
        return predictor
    except Exception as e:
        logger.error(f"Error initializing predictor: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model initialization error: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(customer_data: CustomerData, predictor: ChurnPredictor = Depends(get_predictor)):
    try:
        request_id = str(uuid.uuid4())
        data_dict = customer_data.dict()
        result = predictor.predict(data_dict)
        explanation = predictor.explain_prediction(data_dict)
        explanation_list = [{"feature": feat, "importance": float(imp)} for feat, imp in explanation] if explanation else None

        response = {
            "churn_probability": float(result["churn_probability"][0]),
            "churn_prediction": bool(result["churn_prediction"][0]),
            "explanation": explanation_list,
            "request_id": request_id
        }
        logger.info(f"Prediction request {request_id}: Probability {response['churn_probability']:.4f}")
        return response
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    predictor: ChurnPredictor = Depends(get_predictor)
):
    try:
        job_id = str(uuid.uuid4())
        temp_file_path = f"temp_{job_id}_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        batch_jobs[job_id] = {"status": "queued", "file_path": temp_file_path, "result_file": None}
        background_tasks.add_task(process_batch_prediction, job_id, temp_file_path, predictor)

        return {"job_id": job_id, "status": "queued", "message": "Batch prediction job has been queued"}
    except Exception as e:
        logger.error(f"Error scheduling batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Function to process batch predictions
def process_batch_prediction(job_id: str, file_path: str, predictor: ChurnPredictor):
    try:
        batch_jobs[job_id]["status"] = "processing"
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")

        batch_jobs[job_id]["total_records"] = len(data)
        results = predictor.predict(data)
        results_df = pd.DataFrame({
            "churn_probability": results["churn_probability"],
            "churn_prediction": results["churn_prediction"]
        })
        if "customerID" in data.columns:
            results_df["customerID"] = data["customerID"]

        result_file = f"{BATCH_RESULTS_DIR}/batch_prediction_{job_id}.csv"
        results_df.to_csv(result_file, index=False)
        batch_jobs[job_id].update({"status": "completed", "result_file": result_file, "completed_records": len(data)})
        os.remove(file_path)
    except Exception as e:
        logger.error(f"Error in batch prediction job {job_id}: {str(e)}")
        batch_jobs[job_id].update({"status": "failed", "error": str(e)})
        if os.path.exists(file_path):
            os.remove(file_path)

# Batch prediction status endpoint
@app.get("/batch-predict/{job_id}/status", response_model=BatchPredictionStatus)
def get_batch_status(job_id: str):
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = batch_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "total_records": job.get("total_records"),
        "completed_records": job.get("completed_records"),
        "result_file": os.path.basename(job["result_file"]) if job.get("result_file") else None
    }

# Download batch results endpoint
@app.get("/batch-predict/{job_id}/download")
def download_batch_results(job_id: str):
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = batch_jobs[job_id]
    if job["status"] != "completed" or not job.get("result_file"):
        raise HTTPException(status_code=400, detail="Results not available yet")
    result_file = job["result_file"]
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path=result_file, filename=os.path.basename(result_file), media_type="text/csv")

# Root endpoint
@app.get("/")
def root():
    return {
        "name": "Customer Churn Prediction API",
        "version": "1.0.0",
        "description": "API for predicting customer churn using a machine learning model",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API information"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/predict", "method": "POST", "description": "Single customer prediction"},
            {"path": "/batch-predict", "method": "POST", "description": "Batch prediction from file"},
            {"path": "/batch-predict/{job_id}/status", "method": "GET", "description": "Check batch job status"},
            {"path": "/batch-predict/{job_id}/download", "method": "GET", "description": "Download batch results"}
        ]
    }
