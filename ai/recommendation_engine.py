def build_dataset_summary(profile, scan_report):
    summary = f""" Dataset Overview
    Rows: {profile['rows']}
    Columns: {profile['columns']}

    Missing Values: {profile['missing_values']}
    Duplicate Rows: {profile['duplicate_rows']}
    Invalid Dates: {sum(scan_report['invalid_dates'].values())}
    Outliers: {sum(scan_report['extreme_outliers'].values())}
    High Cardinality: {scan_report.get('high_cardinality_columns', [])}
    """
    
    return summary

def build_ai_prompt(summary):
    prompt = f"""
You are a senior enterprise data consultant.

Analyze the dataset.

Return EXACTLY this structure.

EXECUTIVE_SUMMARY:
(write summary here)

KEY_RISKS:
(write risks here)

RECOMMENDED_ACTIONS:
(write actions here)

BUSINESS_OPPORTUNITIES:
(write opportunities here)

DATASET_HEALTH_SCORE:
Provide a score from 0-100

CONFIDENCE_SCORE:
Provide a score from 0-100

Rules:
- Each section must contain exactly 2 bullet points.
- Keep the report under 150 words.
- Do not use markdown headings.
- Do not change section names.
- Avoid repeating information across sections.

Dataset:

{summary}
"""
    return prompt