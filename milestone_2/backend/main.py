from fastapi import FastAPI
import joblib
import numpy as np
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

app = FastAPI()

model = joblib.load("modellog.joblib")
scaler = joblib.load("minmaxscaler.joblib")

FEATURE_ORDER = [
"Account length",
"Area code",
"International plan",
"Voice mail plan",
"Number vmail messages",
"Total day minutes",
"Total day calls",
"Total day charge",
"Total eve minutes",
"Total eve calls",
"Total eve charge",
"Total night minutes",
"Total night calls",
"Total night charge",
"Total intl minutes",
"Total intl calls",
"Total intl charge",
"Customer service calls"
]

@app.post("/predict")
def predict(data: dict):
    features = [data[col] for col in FEATURE_ORDER]
    features = np.array([features])
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    return {
        "churn": int(pred),
        "probability": float(prob)
    }