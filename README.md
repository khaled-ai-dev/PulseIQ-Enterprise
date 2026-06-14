# PulseIQ Enterprise
**AI-Powered Data Intelligence Platform**

PulseIQ Enterprise automatically profiles business datasets, detects data quality 
issues, generates executive reports, and builds machine learning models — all from 
a single upload. Built with Python, Streamlit, and Llama 3.

---

## What It Does

Upload any business dataset and PulseIQ will:

- Score your data quality from 0 to 100
- Detect missing values, duplicates, outliers, and formatting errors
- Generate an executive summary with key risks and recommendations
- Train a machine learning model and return predictions
- Identify which variables drive your outcomes the most
- Standardize inconsistent categories using AI

---

## Features

**Health Score Engine**: 
- Scores overall data quality on a 0–100 scale before any analysis begins.

**Data Quality Assessment**: 
- Flags missing values, duplicate records, invalid dates, outliers, and negative 
financial values.

**Executive Intelligence Reports**: 
Generates business-oriented summaries written for decision makers, not just 
data teams.

**AI-Powered Recommendations**: 
Uses Llama 3 to produce specific, context-aware recommendations for improving 
data quality and analytical outcomes.

**Predictive Analytics**: 
Automatically detects the target variable, trains a Random Forest model, and 
returns predictions with full performance metrics.

**Feature Importance Analysis**: 
Ranks variables by their influence on the prediction outcome.

**Smart Category Standardization**: 
Detects and merges inconsistent categorical values using AI — for example, 
recognizing that "Full Time", "full-time", and "FT" refer to the same category.

---

## System Architecture

```
Dataset Upload
↓
Automated Data Profiling
↓
Data Quality Assessment
↓
Health Score Calculation
↓
AI Recommendation Engine (Llama 3)
↓
Feature Importance Analysis
↓
Predictive Model Generation
↓
Executive Intelligence Dashboard
```

## Technologies

### Artificial Intelligence
- Llama 3
- Ollama

### Machine Learning
- Random Forest Classifier
- Random Forest Regressor
- Label Encoding

### Data Processing
- Pandas
- NumPy

### Visualization
- Streamlit

### Development
- Python
- Git
- GitHub

---

## Business Value

Data quality problems are rarely visible until they cause a real mistake — a wrong forecast, a misleading report, or a model that performs well in testing and fails in production.

PulseIQ was built to surface those problems before they matter. By automating the profiling, assessment, and reporting process, it gives teams a clear picture of their data's reliability in minutes rather than hours.

The result is faster analysis, more confident decision making, and fewer surprises downstream.
