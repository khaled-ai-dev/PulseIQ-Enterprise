import pandas as pd

from sklearn.ensemble import(RandomForestClassifier)
from sklearn.ensemble import (RandomForestRegressor)
from sklearn.preprocessing import (LabelEncoder)

def generate_feature_importance(df, target_column):
    if target_column not in df.columns:
        return None
    temp_df = df.copy()
    temp_df = temp_df.dropna()

    if len(temp_df) < 20:
        return None
    
    id_columns = []
    for col in df.columns:
        col_lower = col.lower()
        uniqueness_ratio = df[col].nunique() / len(df)

        if("id" in col_lower
           or "uuid" in col_lower
           or "key" in col_lower
           or uniqueness_ratio > 0.95):
            id_columns.append(col)

    X = temp_df.drop(columns=[target_column] + id_columns, errors="ignore")
    y = temp_df[target_column]

    if len(temp_df) > 5000:
        temp_df = temp_df.sample(5000, random_state=42)
        X = X.loc[temp_df.index]
        y = y.loc[temp_df.index] if hasattr(y, "loc") else y[temp_df.index]

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    model.fit(X,y)

    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    return importance_df.head(5)