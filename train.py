import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("Loading data from DVC...")
# 1. Load the generated dataset
df = pd.read_csv("data/mtsamples.csv")

# Because you used the generator script, your columns are already perfectly named!
X = df['text']
y = df['severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Set up MLflow Tracking
# This tells MLflow to create a new experiment
mlflow.set_experiment("TriageOS_Severity_Classifier")

with mlflow.start_run():
    print("Training the NLP model...")
    
    # Define our model parameters
    max_features = 1000
    C_param = 1.0

    # Create an NLP pipeline: Converts text to numbers (TF-IDF), then trains a classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
        ('clf', LogisticRegression(C=C_param, max_iter=200, class_weight='balanced'))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # 3. Log everything to MLflow (This is the MLOps part!)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("C_parameter", C_param)
    mlflow.log_metric("accuracy", acc)

    # Save the trained model artifact
    mlflow.sklearn.log_model(pipeline, "triage_model")

    print("✅ Experiment and model successfully logged to MLflow!")