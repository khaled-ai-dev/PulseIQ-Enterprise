import pandas as pd
import numpy as np

def scan_dataset(df):
    report = {"invalid_dates": {}, "negative_finance": {}, "extreme_outliers": {}, "missing_values": {}, "duplicates": 0, "empty_columns": [], "constant_columns": [], "mixed_data_types": {}, "high_cardinality_columns": [], "column_types": {}, "highly_skewed": {}, "high_correlations": [], "inconsistent_categories": {}}

    for col in df.columns:
        date_keywords = ["date", "timestamp", "created_at", "updated_at"]
        if any(keyword in col.lower() for keyword in date_keywords):
            sample = (df[col].dropna().astype(str).head(100))
            date_like_ratio = sample.str.contains(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", regex=True).mean()
            if date_like_ratio < 0.5:
                continue
            converted = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            failures = (converted.isna().sum() - df[col].isna().sum())
            if failures > 0:
                report["invalid_dates"][col] = int(failures)

    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["sales", "revenue", "price", "profit"]):
            try:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    report["negative_finance"][col] = int(neg_count)
            except:
                pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            report["extreme_outliers"][col] = int(outlier_count)

    missing = df.isna().sum()
    for col, count in missing.items():
        if count > 0:
            report["missing_values"][col] = int(count)

    report["duplicates"] = int(df.duplicated().sum())

    for col in df.columns:
        if df[col].isna().all():
            report["empty_columns"].append(col)
    
    for col in df.columns:
        if df[col].nunique(dropna=False) == 1:
            report["constant_columns"].append(col)

    for col in df.columns:
        unique_types = (df[col].dropna().apply(type).nunique())
        if unique_types > 1:
            report["mixed_data_types"][col] = int(unique_types)

    for col in df.columns:
        uniqueness_ratio = (df[col].nunique(dropna=True) / len(df))
        if uniqueness_ratio > 0.95:
            report["high_cardinality_columns"].append({"column": col, "uniqueness_ratio": round(uniqueness_ratio * 100, 2)})

    for col in df.columns:
        report["column_types"][col] = str(df[col].dtype)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        skewness = abs(df[col].skew())
        if skewness > 1:
            report["highly_skewed"][col] = round(skewness, 2)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        corr_matrix = (df[numeric_cols].corr().abs())
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > 0.85:
                    report["high_correlations"].append({"feature_1": corr_matrix.columns[i], "feature_2": corr_matrix.columns[j], "correlation": round(corr_value, 3)})


    for col in df.columns:
        if df[col].dtype == "object":
            values = df[col].dropna().astype(str).str.strip()
            if values.empty:
                continue
            unique_count = values.nunique()
            cleaned = values.str.replace(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", regex=True)
            cleaned = cleaned.str.replace(r"\s*[-–—]+\s*", "-", regex=True)
            cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
            normalized_count = values.str.lower().nunique()
            if unique_count > normalized_count and unique_count <= 50:
                report["inconsistent_categories"][col] = {
                    "raw_unique": int(unique_count),
                    "normalized_unique": int(normalized_count),
                    "sample_values": values.unique()[:10].tolist()
                }

    return report

