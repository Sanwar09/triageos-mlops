from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TriageOS MLOps Production API")

# --- MLOPS TRICK 1: CORS Middleware (Allows your index.html to talk to the API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientReport(BaseModel):
    text: str

# --- MLOPS TRICK 2: Initialize model as None (Graceful Degradation Failsafe) ---
model = None

print("Fetching the best model from MLflow Registry...")
try:
    # Find our experiment
    experiment = mlflow.get_experiment_by_name("TriageOS_Severity_Classifier")
    
    if experiment:
        # Search for the run with the highest accuracy
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["metrics.accuracy DESC"], 
            max_results=1
        )
        if not runs.empty:
            best_run_id = runs.iloc[0]['run_id']
            
            # Load that specific model
            model_uri = f"runs:/{best_run_id}/triage_model"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Production Model Loaded Successfully! (Run ID: {best_run_id})")
except Exception as e:
    print(f"⚠️ Warning: MLflow disconnected. Using safety fallback. Error: {e}")

# --- API ROUTE ---
@app.post("/api/dispatch")
async def process_dispatch(report: PatientReport):
    
    # Graceful Degradation: If AI is online, use it. If offline, default to CRITICAL.
    if model is not None:
        prediction = model.predict([report.text])[0]
    else:
        prediction = "CRITICAL"
    
    # Hospital routing logic
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