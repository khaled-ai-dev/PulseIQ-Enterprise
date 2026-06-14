def build_executive_report(health_score, critical_issues):
    if health_score >= 85:
        grade = "A"
    elif health_score >= 70:
        grade = "B"
    elif health_score >= 50:
        grade = "C"
    else:
        grade = "D"

    if critical_issues == 0:
        risk ="Moderate"
    elif critical_issues < 100:
        risk = "Moderate"
    else:
        risk = "High"

    return {"grade": grade, "risk": risk}