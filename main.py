from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="TriageOS MLOps Production API")

class PatientReport(BaseModel):
    text: str

# --- MLOPS TRICK: Automatically fetch the best model from MLflow ---
print("Fetching the best model from MLflow Registry...")
try:
    # Find our experiment
    experiment = mlflow.get_experiment_by_name("TriageOS_Severity_Classifier")
    
    # Search for the run with the highest accuracy
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["metrics.accuracy DESC"], 
        max_results=1
    )
    best_run_id = runs.iloc[0]['run_id']
    
    # Load that specific model
    model_uri = f"runs:/{best_run_id}/triage_model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Production Model Loaded Successfully! (Run ID: {best_run_id})")
except Exception as e:
    print(f"Error loading model: {e}. Make sure you ran train.py first!")

# --- API ROUTE ---
@app.post("/api/dispatch")
async def process_dispatch(report: PatientReport):
    # Use the local model to predict severity instead of the Gemini API
    prediction = model.predict([report.text])[0]
    
    # Simulate the multi-agent routing logic based on the prediction
    bed_assignment = "TRAUMA-BAY" if prediction == "CRITICAL" else "STANDARD-ER"
    
    return {
        "status": "success",
        "input_data": report.text,
        "ai_triage_result": {
            "severity_class": prediction,
            "assigned_bed": bed_assignment,
            "action": "Hospital staff alerted via MLOps pipeline"
        }
    }

@app.get("/")
async def root():
    return {"message": "TriageOS MLOps API is running!"}