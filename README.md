# AI Chatbot & Customer Churn Prediction System

## Overview
An end-to-end intelligent platform combining conversational AI with machine learning-based customer churn prediction. The entire system is orchestrated via **n8n** workflows to split incoming user requests into either an intelligent chat path or an ML prediction path.

## Technology Stack
- **Frontend:** HTML / CSS / JavaScript
- **Backend:** FastAPI (Python) + ML Models (deployed on Railway)
- **AI & Automation:** Groq LLM, n8n Workflows
- **Notification:** Gmail API

---

## Core Workflows

The system processes requests via a unified webhook and branches into two main pipelines:

### 1. Chatbot Flow
A two-tier intelligent answering system:
- **Tier 1 (Speed):** Attempts keyword-based matching against a pre-loaded FAQ dataset.
- **Tier 2 (Intelligence):** Falls back to **Groq LLM** for context-aware, generated answers if no exact FAQ matches are found.

### 2. Prediction Flow
A machine-learning pipeline for active predictions:
- **Compute:** Routes customer data to the **FastAPI backend**.
- **Evaluate:** Model 1 computes **Churn Probability**, while Model 2 extracts **Feature Importance** (key risk factors).
- **Explain:** **Groq LLM** interprets the ML outputs and translates them into a plain-English explanation.
- **Notify:** Generates the final UI response and emails a comprehensive prediction report to the user via the **Gmail API**.

---
---

## Milestone 1 (Previous Reference)

- **Goal:** Predict telecom customer churn using **Logistic Regression** (`class_weight="balanced"`).
- **Data:** `churn-bigml-20.csv` (Contains significant class imbalance with only ~14.2% churn).
- **Evaluation Strategy:** Focused on **Recall** and **ROC-AUC** rather than raw accuracy to avoid missing real churners. The prediction threshold was intentionally lowered to **0.3** to optimize recall.
- **Saved Assets:** Both the trained model (`modellog.joblib`) and the data scaler (`minmaxscaler.joblib`) are persisted.
- **Dashboard:** Initially built as an interactive dashboard using Streamlit (`app/streamlit_app.py`) to explore dataset features, toggle threshold parameters, and display live confusion matrices/ROC curves.
