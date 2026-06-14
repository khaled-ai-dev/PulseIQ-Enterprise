import streamlit as st
import pandas as pd
import re

from utils.dataset_scanner import scan_dataset
from utils.interactive_cleaner import (clean_dataset_interactive)
from analytics.profile_engine import(generate_profile)
from analytics.health_score import(calculate_health_score)
from analytics.executive_report import(build_executive_report)
from analytics.target_detector import(suggest_target_column)
from analytics.feature_importance import(generate_feature_importance)
from analytics.business_analyzer import(analyze_business_value)
from analytics.predictive_model import(train_prediction_model)
from ai.ai_consultant import(generate_ai_recommendations, generate_category_mapping)
from ai.recommendation_engine import(build_dataset_summary, build_ai_prompt)

st.set_page_config(page_title="PulseIQ", page_icon="📊", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background-color:#0f172a;
}
            
[data-testid="stSidebar"]{
    background-color:#111827;
}

h1,h2,h3,h4,p,label{
    color:white;            
}

[data-testid="stMetric"]{
    background:#111827;
    border-radius:12px;
    padding: 15px;
    border:1px solid #374151;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
                        
.main {
    padding-top: 1rem;     
}
            
[data-testid="stMetric"] {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;      
    border: 1px solid #374151;        
}       

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color:#111827;
    border-radius:15px;
    padding:20px;
    border:1px solid #374151
} 
            
h1 {
    color: white;            
}

h2 {
    color: #60A5FA;
}            

</style>            
""", unsafe_allow_html=True)


st.markdown("""
# PulseIQ Enterprise
### AI-Powered Data Intelligence Platform

Upload any business dataset and receive:
- Data quality assessment
- Automated cleaning recommendations
- Business insights
- AI-powered analytics guidance

---                
""")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    dataset_signature = f"{df.shape[0]}_{df.shape[1]}"
    if st.session_state.get("dataset_signature") != dataset_signature:
        st.session_state["dataset_signature"] = dataset_signature
        st.session_state.pop("ai_report", None)
        st.session_state.pop("cleaned_df", None)
    
    profile = generate_profile(df)
    with st.expander("View Raw Dataset"):
        st.dataframe(df.head(50))

    def get_severity(count):
        if count == 0:
            return "Low"
        elif count < 100:
            return "Medium"
        elif count < 1000:
            return "High"
        
        return "Critical"

    scan_report = scan_dataset(df)
    business_insights = analyze_business_value(profile, scan_report)
    health_score = calculate_health_score(profile, scan_report)
    critical_issues = (sum(scan_report["invalid_dates"].values())
                       + 
                       sum(scan_report["negative_finance"].values())
                       +
                       sum(scan_report["extreme_outliers"].values()))
    executive_report = build_executive_report(health_score, critical_issues)

    st.markdown(f"""
    <div style="
    padding:25px;
    border-radius:15px;
    background:#111827;
    border:1px solid #374151;
    margin-bottom:20px;        
    ">
                
    <h2 style="color:white;">
    PulseIQ Intelligence Report
    </h2>
                
    <p style="color:white;">
    Rows: {profile['rows']} | Columns: {profile['columns']} | Health Score: {health_score}/100
    </p>

    </div>
""", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="
    padding:25px;
    border-radius:15px;
    background:#111827;
    border:1px solid #374151;
    margin-bottom:20px;
    ">
    
    <h2 style="color:#60A5FA;">
    Executive Intelligence Brief
    </h2>
                
    <h3 style="color:white;">
    Data Quality Grade:
    {executive_report['grade']}
    </h3>

    <p style="color:white;">
    Business Risk Level:
    {executive_report['risk']}
    </p>

    <p style="color:white;">
    Critical Issues:
    {critical_issues}
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("Excutive Dashboard")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Health Score", f"{health_score}/100")
    with col2:
        st.metric("Critical Issues", critical_issues)
    with col3:
        st.metric("Rows", profile["rows"])
    with col4:
        st.metric("Columns", profile["columns"])


    st.subheader("Executive Summary")
    if health_score >= 85:
        badge = "🟢 Excellent"
    elif health_score >= 70:
        badge = "🟡 Good"
    elif health_score >= 50:
        badge = "🟠 Moderate"
    else:
        badge = "🔴 Critical"
    
    st.markdown(f"""
    ### Dataset Health Status
                
    {badge}
    
    Current Health Score: **{health_score}/100**
    """)


    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Missing Values", profile["missing_values"])
    with col2:
        st.metric("Duplicates", profile["duplicate_rows"])
    with col3:
        st.metric("Numeric Columns", profile["numeric_columns"])
    with col4:
        st.metric("Memory (MB)", profile["memory_usage_mb"])

    st.subheader("Data Quality Report")
    total_invalid_dates = sum(scan_report["invalid_dates"].values())
    total_negative_finance = sum(scan_report["negative_finance"].values())
    total_outliers = sum(scan_report["extreme_outliers"].values())
    total_missing = sum(scan_report["missing_values"].values())
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Invalid Dates", total_invalid_dates)

    with col2:
        st.metric("Negative Finance Records", total_negative_finance)
    
    with col3:
        st.metric("Outliers", total_outliers)

    with col4:
        st.metric("Missing Values", total_missing)

    st.subheader("Detailed Data Quality Findings")
    tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Empty Columns", "Constant Columns", "Mixed Types"])
    with tab1:
        if scan_report["missing_values"]:
            missing_df = pd.DataFrame(scan_report["missing_values"].items(), columns=["Column", "Missing Count"])
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No Missing Values Detected")

    with tab2:
        if scan_report["empty_columns"]:
            empty_df = pd.DataFrame(scan_report["empty_columns"], columns=["Column"])
            st.dataframe(empty_df, use_container_width=True)
        else:
            st.success("No Empty Columns Detcted")

    with tab3:
        if scan_report["constant_columns"]:
            constant_df = pd.DataFrame(scan_report["constant_columns"], columns=["Column"])
            st.dataframe(constant_df, use_container_width=True)
        else:
            st.success("No Constant Columns Detected")
    
    with tab4:
        if scan_report["mixed_data_types"]:
            mixed_df = pd.DataFrame(scan_report["mixed_data_types"].items(), columns=["Column", "Type Count"])
            st.dataframe(mixed_df, use_container_width=True)
        else:
            st.success("No Mixed Types Detected")


    st.subheader("Issue Severity")
    if total_invalid_dates > 0:
        st.warning(f"Invalid Dates: {total_invalid_dates} ({get_severity(total_invalid_dates)})")
    if total_negative_finance > 0:
        st.warning(f"Negative Finance Records: {total_negative_finance} ({get_severity(total_negative_finance)})")
    if total_outliers > 0:
        st.warning(f"Outliers: {total_outliers} ({get_severity(total_outliers)})")
    if (total_invalid_dates == 0 and total_negative_finance == 0 and total_outliers == 0):
        st.success("No Major Issues Detected.")

    critical_issues = (total_invalid_dates + total_negative_finance + total_outliers)
    st.metric("Critical Issues", critical_issues)
    
    st.subheader("AI Data Scientist Review")
    st.metric("AI Readiness Score", f"{health_score}/100")
    if health_score >= 85:
        st.success("High AI Readiness")
    elif health_score >= 70:
        st.warning("Moderate AI Readiness")
    else:
        st.error("Low AI Readiness")

    summary = build_dataset_summary(profile, scan_report)
    prompt = build_ai_prompt(summary)
    if st.button("Generate AI Recommendations"):
        if "ai_report" not in st.session_state:
            with st.spinner("AI is analyzing dataset..."):
                ai_response = generate_ai_recommendations(prompt)
                st.session_state["ai_report"] = ai_response
                    

    if "ai_report" in st.session_state:
        ai_response = st.session_state["ai_report"]
        health_score_ai = re.search(r"DATASET_HEALTH_SCORE:\s*(\d+)", ai_response)
        confidence_score_ai = re.search(r"CONFIDENCE_SCORE:\s*(\d+)", ai_response)

        summary = ""
        risks = ""
        actions = ""
        opportunities = ""

        try:
            summary = re.search(r"EXECUTIVE_SUMMARY:(.*?)KEY_RISKS:", ai_response, re.S).group(1).strip()
            risks = re.search(r"KEY_RISKS:(.*?)RECOMMENDED_ACTIONS:", ai_response, re.S).group(1).strip()
            actions = re.search(r"RECOMMENDED_ACTIONS:(.*?)BUSINESS_OPPORTUNITIES:", ai_response, re.S).group(1).strip()
            opportunities = re.search(r"BUSINESS_OPPORTUNITIES:(.*?)DATASET_HEALTH_SCORE:", ai_response, re.S).group(1).strip()
        except:
            st.error("Unable to parse AI report.")
            
            summary = ai_response
            risks = "Unable to extract"
            actions = "Unable to extract"
            opportunities = "Unable to extract"
        
        summary = summary.replace("•", "\n\n•")
        risks = risks.replace("•", "\n\n•")
        actions = actions.replace("•", "\n\n•")
        opportunities = opportunities.replace("•", "\n\n•")

        st.markdown("## AI Executive Report")
        metric1, metric2 = st.columns(2)
        with metric1:
            if health_score_ai:
                st.metric("AI Dataset Health Score", health_score_ai.group(1))
        
        with metric2:
            if confidence_score_ai:
                st.metric("AI Confidence Score", confidence_score_ai.group(1) + "%")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 📋 Executive Summary")
                st.markdown(summary)

        with col2:
            with st.container(border=True):
                st.markdown("### ⚠️ Key Risks")
                st.markdown(risks)

        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown("### ✅ Recommended Actions")
                st.markdown(actions)

        with col4:
            with st.container(border=True):
                st.markdown("### 🚀 Business Opportunities")
                st.markdown(opportunities)
            
        
        if st.button("Clear AI Report"):
            del st.session_state["ai_report"]
            st.rerun()

    with st.expander("Intelligent Cleaning Controls", expanded=False):
        st.subheader("Categorical Normalization")
        category_mappings = {}
        if scan_report["inconsistent_categories"]:
            st.write("These columns have inconsistent values that may represent the same category:")
            for col, info in scan_report["inconsistent_categories"].items():
                st.write(f"**{col}**: {info['raw_unique']} raw values → {info['normalized_unique']} after basic normalization")
                st.write(f"Sample values: {info['sample_values']}")
                use_ai = st.checkbox(f"Use AI to clean {col}", value=False, key=f"{col}_ai_clean")
                if use_ai:
                    unique_vals = working_series_unique = df[col].dropna().astype(str).str.strip().unique().tolist()
                    cache_key = f"{col}_ai_mapping"
                    if cache_key not in st.session_state:
                        with st.spinner(f"AI is analyzing {col}..."):
                            mapping = generate_category_mapping(col, unique_vals)
                            st.session_state[cache_key] = mapping
                    mapping = st.session_state[cache_key]
                    if mapping:
                        category_mappings[col] = mapping
                        st.success(f"AI generated {len(set(mapping.values()))} canonical categories for {col}")
                    else:
                        st.warning(f"AI mapping failed for {col}, will use basic normalization instead")
                        category_mappings[col] = None
                else:
                    category_mappings[col] = None
        else:
            st.success("No inconsistent categorical values detected")



        if scan_report["invalid_dates"]:
            st.subheader("Invalid Date Handling")
        else:
            st.success("No invalid dates found")
        date_choices = {}
        date_fill_methods = {}
        for col in scan_report["invalid_dates"]:
            choice = st.selectbox(f"What should we do with invalid dates in {col}?",
                ["drop_rows", "fill_placeholder", "leave_as_nan"], key=f"{col}_main_choice")
            date_choices[col] = choice
            if choice == "fill_placeholder":
                fill_method = st.selectbox(f"Choose fill strategy for {col}",
                    ["median_date", "forward_fill", "backward_fill"], key=f"{col}_fill_method")
                date_fill_methods[col] = fill_method

        st.subheader("Negative Finance Handling")
        finance_choices = {}
        for col in scan_report["negative_finance"]:
            choice = st.selectbox(f"What should we do with negative values in {col}?",
                ["convert_to_absolute", "cap_at_zero", "convert_to_nan", "drop_rows"])
            finance_choices[col] = choice

        st.subheader("Outlier Handling")
        outlier_choices = {}
        for col in scan_report["extreme_outliers"]:
            choice = st.selectbox(f"What should we do with outliers in {col}?",
                ["cap_values_winsorize", "log_transform", "flag_only", "drop_rows"])
            outlier_choices[col] = choice

        user_choices = {
            "Category_Mappings": category_mappings,
            "Invalid_Date_Handling": date_choices,
            "date_fill_methods": date_fill_methods,
            "Handling_Negative_Finance": finance_choices,
            "Handling_Outliers": outlier_choices
        }
        st.session_state["user_choices"] = user_choices

    if st.button("Run Intelligent Cleaning"):
        if "user_choices" in st.session_state:
            cleaned_df = clean_dataset_interactive(df, st.session_state["user_choices"])
            st.session_state["cleaned_df"] = cleaned_df

    if "cleaned_df" in st.session_state:
        cleaned_df = st.session_state["cleaned_df"]
        st.subheader("Cleaned Dataset")
        st.dataframe(cleaned_df.head())
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows Before", df.shape[0])
        with c2:
            st.metric("Rows After", cleaned_df.shape[0])
        with c3:
            st.metric("Columns After", cleaned_df.shape[1])
        working_df = cleaned_df
    else:
        st.info("Run Intelligent Cleaning above to unlock target detection, feature importance and the AI decision simulator.")
        working_df = None

    if working_df is not None:
        target_column, target_scores = suggest_target_column(working_df)
        score_df = pd.DataFrame(target_scores.items(), columns=["Column", "Target Score"])
        score_df = score_df.sort_values(by="Target Score", ascending=False)
        st.subheader("Target Detection Analysis")
        st.dataframe(score_df, use_container_width=True)
        st.success(f"🤖 AI Recommended Target Column: {target_column}")
        target_column = st.selectbox("Choose Target Column", working_df.columns, index=list(working_df.columns).index(target_column))
        feature_importance = generate_feature_importance(working_df, target_column)

        top_drivers = []
        if feature_importance is not None:
            feature_importance["Importance"] = (feature_importance["Importance"] * 100).round(2)

            st.subheader("Feature Importance Analysis")
            st.markdown("### 🏆 Top Predictive Drivers")
            top5 = feature_importance.head(5)
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            for idx, (_, row) in enumerate(top5.iterrows()):
                st.markdown(
                    f"{medals[idx]} "
                    f"**{row['Feature']}** "
                    f"({row['Importance']:.2f}%)"
                )

            st.success(f"Target Column Detected: {target_column}")
            st.markdown("### 📊 Feature Rankings")
            st.dataframe(feature_importance.style.format({"Importance": "{:.2f}%"}), use_container_width=True)

            if not feature_importance.empty:
                top_feature = feature_importance.iloc[0]["Feature"]
                st.info(f"The strongest predictor of {target_column} is {top_feature}.")

            top_drivers = feature_importance.head(3)["Feature"].tolist()

        st.subheader("Predictive AI Engine")
        model, feature_names, encoders, positive_class_index = train_prediction_model(working_df, target_column)
        st.success(f"AI model trained to predict {target_column}")

        X_full = working_df.copy()
        for col in feature_names:
            if col in encoders:
                X_full[col] = X_full[col].astype(str).map(
                    lambda v, le=encoders[col]: le.transform([v])[0] if v in le.classes_ else -1
                )
            elif X_full[col].dtype == "object":
                extracted = X_full[col].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
                X_full[col] = pd.to_numeric(extracted, errors="coerce")

        X_full = X_full[feature_names].fillna(0)
        proba_full = model.predict_proba(X_full)[:, positive_class_index] * 100
        working_df["Risk_Score"] = proba_full

        st.markdown("---")
        st.markdown("## 🏢 Risk by Segment")

        candidate_segments = []
        for col in working_df.columns:
            if col in (target_column, "Risk_Score"):
                continue
            if working_df[col].dtype == "object" or working_df[col].nunique() <= 15:
                if 2 <= working_df[col].nunique() <= 30:
                    candidate_segments.append(col)

        if not candidate_segments:
            st.warning("No suitable categorical column found for segmentation.")
        else:
            segment_col = st.selectbox("Group by", candidate_segments, index=0)

            segment_avg = (
                working_df.groupby(segment_col)["Risk_Score"]
                .mean()
                .reset_index()
                .sort_values("Risk_Score", ascending=False)
            )
            segment_avg["Risk_Score"] = segment_avg["Risk_Score"].round(2)

            highest = segment_avg.iloc[0]
            lowest = segment_avg.iloc[-1]

            st.markdown(
                f"**{highest[segment_col]}** has the highest average predicted **{target_column}** "
                f"risk at **{highest['Risk_Score']:.1f}%**, compared to **{lowest['Risk_Score']:.1f}%** "
                f"in **{lowest[segment_col]}**."
            )

            def bar_color(pct):
                if pct >= 70:
                    return "#EF4444"
                if pct >= 40:
                    return "#F59E0B"
                return "#22C55E"

            max_risk = segment_avg["Risk_Score"].max()
            rows_html = ""
            for _, row in segment_avg.iterrows():
                bar_width = (row["Risk_Score"] / max_risk * 100) if max_risk > 0 else 0
                rows_html += (
                    f'<div style="margin-bottom:14px;">'
                    f'<div style="display:flex; justify-content:space-between; margin-bottom:4px;">'
                    f'<span style="color:white;">{row[segment_col]}</span>'
                    f'<span style="color:{bar_color(row["Risk_Score"])}; font-weight:bold;">{row["Risk_Score"]:.1f}%</span>'
                    f'</div>'
                    f'<div style="background:#1F2937; border-radius:6px; height:10px; width:100%;">'
                    f'<div style="background:{bar_color(row["Risk_Score"])}; width:{bar_width}%; height:10px; border-radius:6px;"></div>'
                    f'</div>'
                    f'</div>'
                )

            html_block = f'<div style="background:#111827; border:1px solid #374151; border-radius:15px; padding:25px; margin-top:15px;">{rows_html}</div>'
            st.markdown(html_block, unsafe_allow_html=True)