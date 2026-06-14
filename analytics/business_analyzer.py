def analyze_business_value(profile, scan_report):
    insights = []
    rows = profile["rows"]
    
    if rows > 10000:
        insights.append("Dataset size is suitable for advanced machine learning.")
    if scan_report["missing_values"]:
        insights.append("Data quality improvements are recommended before model training.")
    if scan_report["high_correlations"]:
        insights.append("Feature reduction opportunities detected.")
    if scan_report["highly_skewed"]:
        insights.append("Feature transformation opportunities detected.")
    if scan_report["duplicates"] > 0:
        insights.append("Duplicate records may impact model accuracy.")
    
    return insights