import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_prediction_model(df, target_column):
    model_df = df.copy()

    if len(model_df) > 5000:
        model_df = model_df.sample(5000, random_state=42)

    encoders = {}

    for col in model_df.columns:
        if model_df[col].dtype == "object":
            cleaned = model_df[col].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
            if cleaned.notna().all():
                model_df[col] = cleaned.astype(float)
            else:
                le = LabelEncoder()
                model_df[col] = le.fit_transform(model_df[col].astype(str))
                encoders[col] = le
    
    X = model_df.drop(columns=[target_column])
    y = model_df[target_column]

    model = RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X,y)

    positive_class_index = 1
    if target_column in encoders:
        classes = list(encoders[target_column].classes_)
        risk_keywords = ["yes", "true", "1", "left", "churn", "default"]
        for i, c in enumerate(classes):
            if str(c).lower() in risk_keywords:
                positive_class_index = i
                break
        else:
            positive_class_index = len(classes) -1
    
    return model, X.columns.tolist(), encoders, positive_class_index