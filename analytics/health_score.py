def calculate_health_score(profile, scan_report):
    score = 100
    rows = profile["rows"]
    
    if rows == 0:
        return 0
    
    missing_values = profile["missing_values"]
    duplicate_rows = profile["duplicate_rows"]

    invalid_dates = sum(scan_report["invalid_dates"].values())
    negative_finance = sum(scan_report["negative_finance"].values())
    outliers = sum(scan_report["extreme_outliers"].values())

    missing_pct = (missing_values / rows) * 100
    duplicate_pct = (duplicate_rows / rows) * 100
    invalid_pct = (invalid_dates / rows) * 100
    outlier_pct = (outliers / rows) * 100

    score -= missing_pct * 0.8
    score -= duplicate_pct * 0.1
    score -= invalid_pct * 0.5
    score -= outlier_pct * 0.3

    if negative_finance > 0:
        score -=5
    
    return max(min(round(score, 1), 100), 0)