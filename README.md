# Churn Prediction — ML Project

This project predicts whether a telecom customer is going to leave (churn) based on their usage patterns. It was built as part of learning about machine learning, class imbalance, and model evaluation.

---

## Dataset

- **Source:** `data/raw/churn-bigml-20.csv`
- **Size:** 667 customers, 20 columns
- **Target:** `Churn` — whether the customer left or not
- **Imbalance:** About **14.2% churned**, 85.8% didn't

This is a small dataset but a real problem. Most customers don't churn, so the model needs to be careful not to ignore the minority group.

---

## Why Accuracy Alone Is Misleading

If a model just predicted "No churn" for everyone, it would get about **85% accuracy**.

That sounds great, but it would miss every single customer who was about to leave — which defeats the whole purpose.

So instead of just looking at accuracy, we focus on:

- **Recall** — out of all customers who actually churned, how many did we catch?
- **ROC-AUC** — how well the model separates churners from non-churners overall

Recall matters most here because missing a churner means losing a customer.

---

## Why We Tuned the Threshold

By default, the model predicts "Churn" only when it's more than 50% sure.

We lowered this to **0.3** — so if the model thinks there's a 30% or higher chance a customer will churn, we flag them.

This increases recall (we catch more churners) but also increases false alarms slightly. It's a tradeoff, and the app lets you adjust the threshold live to see the effect.

---

## Why We Saved the Model AND the Scaler

After training, we saved two files:

```python
joblib.dump(log_model, "models/modellog.joblib")
joblib.dump(scaler,    "models/minmaxscaler.joblib")
```

**The model** is saved so we don't have to retrain every time we want to make a prediction.

**The scaler** is saved because the model was trained on scaled data. If we scale a new customer's data differently (different min/max values), the model will produce wrong predictions. The scaler and model are a pair — they always go together.

---

## How to Run the App

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

The app has three pages:
- **Data Overview** — explore the dataset
- **Model Performance** — see accuracy, recall, confusion matrix, ROC curve (with live threshold slider)
- **Predict Single Customer** — enter customer details and get a risk prediction

---

## How to Load the Model in Python

```python
import joblib

model  = joblib.load("models/modellog.joblib")
scaler = joblib.load("models/minmaxscaler.joblib")

# Example: scale new data, then predict
X_scaled = scaler.transform(X_new)
prediction = model.predict(X_scaled)
probability = model.predict_proba(X_scaled)[:, 1]
```

---

## Project Structure

```
project-root/
├── app/
│   └── streamlit_app.py       ← Streamlit dashboard
├── data/
│   └── raw/
│       └── churn-bigml-20.csv ← original dataset
├── models/
│   ├── modellog.joblib        ← trained logistic regression
│   └── minmaxscaler.joblib    ← fitted scaler
├── notebooks/
│   └── churn_analysis.ipynb   ← full EDA + training notebook
├── train_model.py             ← script to retrain the model
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Model Used

**Logistic Regression** with `class_weight="balanced"` — this tells the model to pay extra attention to churners even though there are fewer of them.

No pipelines or grid search was used — just simple, readable steps so everything is easy to follow.
