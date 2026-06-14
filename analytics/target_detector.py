import pandas as pd

def suggest_target_column(df):
    scores = {}
    high_priority_keywords = ["attrition", "leave", "left", "churn", "turnover", "retention", "default", "fraud"]
    medium_priority_keywords = ["sales", "revenue", "profit", "income", "price", "cost", "purchase"]
    low_priority_keywords = ["target", "label", "outcome", "rating", "score", "satisfaction", "performance"]

    for col in df.columns:
        score = 0
        col_lower = col.lower()
        unique_count = df[col].nunique(dropna=True)
        uniqueness_ratio = df[col].nunique() / len(df)
        missing_ratio = df[col].isna().mean()

        if any(word in col_lower for word in high_priority_keywords):
            score += 60
        elif any(word in col_lower for word in medium_priority_keywords):
            score += 45
        elif any(word in col_lower for word in low_priority_keywords):
            score += 30

        if uniqueness_ratio > 0.95:
            score -= 100
        
        if "id" in col_lower:
            score -= 100

        if "date" in col_lower:
            score -= 50

        if "time" in col_lower:
            score -= 50

        if missing_ratio > 0.50:
            score -= 20

        if unique_count <= 10:
            score += 25

        elif 2 <= unique_count <= 20:
            score += 20

        elif unique_count > 20:
            score += 10

        if pd.api.types.is_numeric_dtype(df[col]):
            score += 15

        scores[col] = score

    highest_score = max(scores.values())
    best_candidates = [col for col, value in scores.items() if value == highest_score]
    best_column = sorted(best_candidates, key=lambda x: df[x].nunique())[0]

    return best_column, scores
    