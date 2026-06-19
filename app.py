import streamlit as st
import pandas as pd
import re

from utils.dataset_scanner import scan_dataset
from utils.interactive_cleaner import (clean_dataset_interactive)
from utils.interactive_cleaner import (clean_category_text)
from analytics.profile_engine import(generate_profile)
from analytics.health_score import(calculate_health_score)
from analytics.executive_report import(build_executive_report)
from analytics.target_detector import(suggest_target_column)
from analytics.feature_importance import(generate_feature_importance)
from analytics.business_analyzer import(analyze_business_value)
from analytics.predictive_model import(train_prediction_model)
from ai.ai_consultant import(generate_ai_recommendations, generate_category_mapping)
from ai.recommendation_engine import(build_dataset_summary, build_ai_prompt)

st.set_page_config(page_title="PulseIQ", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background:#0b1220;
}

[data-testid="stSidebar"]{
    background:#111827;
}

section.main{
    padding-top:0.5rem;
}

.hero{
    background:linear-gradient(135deg,#111827,#1e293b);
    border:1px solid #334155;
    border-radius:20px;
    padding:30px;
    margin-bottom:25px;
}

.hero-title{
    font-size:42px;
    font-weight:700;
    color:white;
}

.hero-subtitle{
    color:#94a3b8;
    font-size:18px;
}

.dashboard-card{
    background:#111827;
    border:1px solid #334155;
    border-radius:18px;
    padding:20px;
}

[data-testid="stMetric"]{
    background:#111827;
    border:1px solid #334155;
    border-radius:18px;
    padding:18px;
}

div[data-testid="stVerticalBlockBorderWrapper"]{
    background:#111827;
    border-radius:18px;
    border:1px solid #334155;
}
            
button[data-baseweb="tab"] {
    font-size: 1.05rem !important;
    flex: 1 !important;
    justify-content: center !important;
}
            
[data-testod="stTabs"] [role="tablist"] {
    display: flex !important;
    width: 100% !important;
}
            
[data-testid="stSidebar"] {
    display: none !important;
}
            
[data-testid="collapsedControl"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.hero-new {
    background: linear-gradient(135deg, #0f1f3d, #1e293b, #0b1220);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero-glow-1 {
    position: absolute;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: #3b82f6;
    opacity: 0.07;
    top: -100px;
    right: -80px;
}
.hero-glow-2 {
    position: absolute;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: #6366f1;
    opacity: 0.06;
    bottom: -80px;
    left: 200px;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 12px;
}
.hero-title-new {
    font-size: 46px;
    font-weight: 800;
    color: white;
    line-height: 1.1;
    margin-bottom: 10px;
}
.hero-title-new span {
    background: linear-gradient(90deg, #60a5fa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle-new {
    color: #64748b;
    font-size: 15px;
    letter-spacing: 0.5px;
}
.hero-pills {
    display: flex;
    gap: 10px;
    margin-top: 20px;
}
.hero-pill {
    background: #1e3a5f;
    border: 1px solid #2d5a8e;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    color: #93c5fd;
    font-weight: 500;
}
</style>

<div class="hero-new">
    <div class="hero-glow-1"></div>
    <div class="hero-glow-2"></div>
    <div class="hero-eyebrow">Enterprise Intelligence Platform</div>
    <div class="hero-title-new">Pulse<span>IQ</span></div>
    <div class="hero-subtitle-new">AI-Powered Data Intelligence — Clean, Analyze, Predict</div>
    <div class="hero-pills">
        <div class="hero-pill">📊 Data Profiling</div>
        <div class="hero-pill">🧹 Smart Cleaning</div>
        <div class="hero-pill">🤖 AI Insights</div>
        <div class="hero-pill">📈 Predictive ML</div>
    </div>
</div>
""", unsafe_allow_html=True)


with st.container(border=True):
    st.subheader("📁 Dataset Upload")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is None:
        st.info("Please Upload a CSV Dataset To begin.")
        st.stop()
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        dataset_signature = f"{df.shape[0]}_{df.shape[1]}"
        if st.session_state.get("dataset_signature") != dataset_signature:
            st.session_state["dataset_signature"] = dataset_signature
            st.session_state.pop("ai_report", None)
            st.session_state.pop("cleaned_df", None)
    

    @st.cache_data
    def cached_profile(df):
        return generate_profile(df)
    
    @st.cache_data
    def cached_scan(df):
        return scan_dataset(df)
    
    @st.cache_data
    def cached_business(profile_json, scan_json):
        return analyze_business_value(profile_json, scan_json)
    
    profile = cached_profile(df)
    scan_report = cached_scan(df)
    business_insights = cached_business(profile, scan_report)


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

    health_score = calculate_health_score(profile, scan_report)
    critical_issues = (sum(scan_report["invalid_dates"].values())
                       + 
                       sum(scan_report["negative_finance"].values())
                       +
                       sum(scan_report["extreme_outliers"].values()))
    executive_report = build_executive_report(health_score, critical_issues)

    
    if health_score >= 85:
        status_color = "#4ade80"
        status_background = "#1a2e1a"
        status_label = "Excellent"
    elif health_score >= 70:
        status_color = "#facc15"
        status_background = "#2e2a1a"
        status_label = "Good"
    elif health_score >= 50:
        status_color = "#fb923c"
        status_background = "#2e1f1a"
        status_label = "Moderate"
    else:
        status_color = "#f87171"
        status_background = "#3b1f1f"
        status_label = "Poor Qualtiy"

    st.markdown(f"""
    <style>
    .cards-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 16px;
        margin-bottom: 16px;
    }}
    .intel-strip {{
        display: flex;
        align-items: center;
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 18px 28px;
        margin-bottom: 20px;
    }}
    </style>

    <div class="cards-grid">
        <div style="background:linear-gradient(135deg,#1e3a5f,#1e293b);border:1px solid #2d5a8e;border-radius:16px;padding:24px 28px;position:relative;overflow:hidden;">
            <div style="position:absolute;width:120px;height:120px;border-radius:50%;right:-30px;top:-30px;background:#3b82f6;opacity:0.12;"></div>
            <div style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#60a5fa;margin-bottom:10px;">Data Quality Grade</div>
            <div style="font-size:52px;font-weight:800;color:#93c5fd;line-height:1;margin-bottom:6px;">{executive_report['grade']}</div>
            <div style="font-size:13px;color:#94a3b8;">Quality assessment of your dataset</div>
        </div>
        <div style="background:linear-gradient(135deg,#3b1f1f,#1e293b);border:1px solid #7f3f3f;border-radius:16px;padding:24px 28px;position:relative;overflow:hidden;">
            <div style="position:absolute;width:120px;height:120px;border-radius:50%;right:-30px;top:-30px;background:#ef4444;opacity:0.12;"></div>
            <div style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#f87171;margin-bottom:10px;">Business Risk Level</div>
            <div style="font-size:52px;font-weight:800;color:#fca5a5;line-height:1;margin-bottom:6px;">{executive_report['risk']}</div>
            <div style="font-size:13px;color:#94a3b8;">Exposure to data-driven errors</div>
        </div>
        <div style="background:linear-gradient(135deg,#1a2e1a,#1e293b);border:1px solid #2d6a2d;border-radius:16px;padding:24px 28px;position:relative;overflow:hidden;">
            <div style="position:absolute;width:120px;height:120px;border-radius:50%;right:-30px;top:-30px;background:#22c55e;opacity:0.12;"></div>
            <div style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#4ade80;margin-bottom:10px;">Critical Issues</div>
            <div style="font-size:52px;font-weight:800;color:#86efac;line-height:1;margin-bottom:6px;">{critical_issues}</div>
            <div style="font-size:13px;color:#94a3b8;">Blocking issues requiring attention</div>
        </div>
    </div>

    <div class="intel-strip">
        <div style="flex:1;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Dataset</div>
            <div style="font-size:18px;font-weight:700;color:#e2e8f0;">PulseIQ Report</div>
        </div>
        <div style="width:1px;height:40px;background:#334155;margin:0 8px;"></div>
        <div style="flex:1;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Rows</div>
            <div style="font-size:22px;font-weight:700;color:#e2e8f0;">{profile['rows']}</div>
        </div>
        <div style="width:1px;height:40px;background:#334155;margin:0 8px;"></div>
        <div style="flex:1;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Columns</div>
            <div style="font-size:22px;font-weight:700;color:#e2e8f0;">{profile['columns']}</div>
        </div>
        <div style="width:1px;height:40px;background:#334155;margin:0 8px;"></div>
        <div style="flex:1;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Health Score</div>
            <div style="font-size:22px;font-weight:700;color:#e2e8f0;">{health_score}<span style="font-size:13px;color:#64748b;">/100</span></div>
        </div>
        <div style="width:1px;height:40px;background:#334155;margin:0 8px;"></div>
        <div style="flex:1;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Status</div>
            <div style="display:inline-block;padding:4px 14px;border-radius:20px;background:{status_background};color:{status_color};font-size:13px;font-weight:600;">{status_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if health_score >= 85:
        hs_color = "#4ade80"
        hs_bg = "linear-gradient(135deg,#1a2e1a,#1e293b)"
        hs_border = "#2d6a2d"
    elif health_score >= 70:
        hs_color = "#facc15"
        hs_bg = "linear-gradient(135deg,#2e2a1a,#1e293b)"
        hs_border = "#6a5a2d"
    elif health_score >= 50:
        hs_color = "#fb923c"
        hs_bg = "linear-gradient(135deg,#2e1f1a,#1e293b)"
        hs_border = "#6a3a2d"
    else:
        hs_color = "#f87171"
        hs_bg = "linear-gradient(135deg,#3b1f1f,#1e293b)"
        hs_border = "#7f3f3f"

    st.markdown(f"""
    <style>
    .dash-grid {{
        display: grid;
        grid-template-columns: 2fr 2fr 2fr 2fr 1.2fr;
        gap: 14px;
        margin-bottom: 14px;
    }}
    .dash-card {{
        border-radius: 16px;
        padding: 22px 24px;
        position: relative;
        overflow: hidden;
    }}
    .dash-card-label {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 10px;
    }}
    .dash-card-value {{
        font-size: 34px;
        font-weight: 800;
        line-height: 1;
    }}
    .btn-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-bottom: 6px;
    }}
    </style>

    <div class="dash-grid">
        <div class="dash-card" style="background:{hs_bg}; border:1px solid {hs_border};">
            <div class="dash-card-label">Health Score</div>
            <div class="dash-card-value" style="color:{hs_color};">{health_score}<span style="font-size:16px;color:#475569;">/100</span></div>
        </div>
        <div class="dash-card" style="background:linear-gradient(135deg,#1e1e3b,#1e293b); border:1px solid #3f3f7f;">
            <div class="dash-card-label">Critical Issues</div>
            <div class="dash-card-value" style="color:#a5b4fc;">{critical_issues}</div>
        </div>
        <div class="dash-card" style="background:linear-gradient(135deg,#1e2e3b,#1e293b); border:1px solid #2d4a6a;">
            <div class="dash-card-label">Rows</div>
            <div class="dash-card-value" style="color:#7dd3fc;">{profile['rows']}</div>
        </div>
        <div class="dash-card" style="background:linear-gradient(135deg,#1e2e3b,#1e293b); border:1px solid #2d4a6a;">
            <div class="dash-card-label">Columns</div>
            <div class="dash-card-value" style="color:#7dd3fc;">{profile['columns']}</div>
        </div>
        <div class="dash-card" style="background:linear-gradient(135deg,#2a1a3b,#1e293b); border:1px solid #5a2d8e;">
            <div class="dash-card-label">Missing Values</div>
            <div class="dash-card-value" style="color:#c4b5fd;">{profile['missing_values']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_button1, col_button2 = st.columns(2)
    with col_button1:
        generate_ai = st.button("🤖 Generate AI Report", use_container_width=True)
    with col_button2:
        run_cleaning = st.button("🧹 Run Cleaning", use_container_width=True)


    overview_tab, quality_tab, cleaning_tab, ai_tab, ml_tab = st.tabs(["Overview", "Data Quality", "Cleaning", "AI", "Machine Learning"])

    with overview_tab:
        if health_score >= 85:
            badge_icon = "🟢"
            badge_text = "Excellent"
            badge_color = "#4ade80"
            badge_bg = "#1a2e1a"
            badge_border = "#2d6a2d"
            bar_color = "#4ade80"
        elif health_score >= 70:
            badge_icon = "🟡"
            badge_text = "Good"
            badge_color = "#facc15"
            badge_bg = "#2e2a1a"
            badge_border = "#6a5a2d"
            bar_color = "#facc15"
        elif health_score >= 50:
            badge_icon = "🟠"
            badge_text = "Moderate"
            badge_color = "#fb923c"
            badge_bg = "#2e1f1a"
            badge_border = "#6a3a2d"
            bar_color = "#fb923c"
        else:
            badge_icon = "🔴"
            badge_text = "Critical"
            badge_color = "#f87171"
            badge_bg = "#3b1f1f"
            badge_border = "#7f3f3f"
            bar_color = "#f87171"

        bar_width = health_score

        st.markdown(f"""
        <style>
        .overview-grid {{
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 14px;
            margin-bottom: 14px;
        }}
        .ov-left {{
            background: linear-gradient(135deg, #0f1f3d, #1e293b);
            border: 1px solid #1e3a5f;
            border-radius: 16px;
            padding: 28px 32px;
            position: relative;
            overflow: hidden;
        }}
        .ov-glow {{
            position: absolute;
            width: 200px; height: 200px;
            border-radius: 50%;
            background: #3b82f6;
            opacity: 0.06;
            top: -60px; right: -60px;
        }}
        .ov-score-label {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            color: #475569;
            margin-bottom: 8px;
        }}
        .ov-score-value {{
            font-size: 64px;
            font-weight: 900;
            color: {badge_color};
            line-height: 1;
            margin-bottom: 6px;
        }}
        .ov-score-sub {{
            font-size: 13px;
            color: #475569;
            margin-bottom: 18px;
        }}
        .ov-bar-track {{
            background: #1e293b;
            border-radius: 99px;
            height: 8px;
            width: 100%;
            margin-bottom: 6px;
            overflow: hidden;
        }}
        .ov-bar-fill {{
            height: 8px;
            border-radius: 99px;
            background: {bar_color};
            width: {bar_width}%;
        }}
        .ov-bar-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #334155;
        }}
        .ov-badge {{
            display: inline-block;
            margin-top: 16px;
            padding: 6px 16px;
            border-radius: 99px;
            background: {badge_bg};
            border: 1px solid {badge_border};
            color: {badge_color};
            font-size: 13px;
            font-weight: 600;
        }}
        .ov-right {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }}
        .ov-stat {{
            background: #111827;
            border-radius: 14px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }}
        .ov-stat-label {{
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #475569;
            margin-bottom: 8px;
        }}
        .ov-stat-value {{
            font-size: 30px;
            font-weight: 800;
            line-height: 1;
        }}
        .ov-stat-dot {{
            position: absolute;
            width: 60px; height: 60px;
            border-radius: 50%;
            opacity: 0.1;
            bottom: -15px; right: -15px;
        }}
        </style>

        <div class="overview-grid">
            <div class="ov-left">
                <div class="ov-glow"></div>
                <div class="ov-score-label">Dataset Health Score</div>
                <div class="ov-score-value">{health_score}</div>
                <div class="ov-score-sub">out of 100 — {badge_text} quality dataset</div>
                <div class="ov-bar-track"><div class="ov-bar-fill"></div></div>
                <div class="ov-bar-labels"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
                <div class="ov-badge">{badge_icon} {badge_text}</div>
            </div>
            <div class="ov-right">
                <div class="ov-stat" style="border:1px solid #2d4a6a;">
                    <div class="ov-stat-dot" style="background:#7dd3fc;"></div>
                    <div class="ov-stat-label">Missing Values</div>
                    <div class="ov-stat-value" style="color:#7dd3fc;">{profile['missing_values']}</div>
                </div>
                <div class="ov-stat" style="border:1px solid #3f3f7f;">
                    <div class="ov-stat-dot" style="background:#a5b4fc;"></div>
                    <div class="ov-stat-label">Duplicates</div>
                    <div class="ov-stat-value" style="color:#a5b4fc;">{profile['duplicate_rows']}</div>
                </div>
                <div class="ov-stat" style="border:1px solid #2d5a8e;">
                    <div class="ov-stat-dot" style="background:#60a5fa;"></div>
                    <div class="ov-stat-label">Numeric Columns</div>
                    <div class="ov-stat-value" style="color:#60a5fa;">{profile['numeric_columns']}</div>
                </div>
                <div class="ov-stat" style="border:1px solid #2d4a3a;">
                    <div class="ov-stat-dot" style="background:#4ade80;"></div>
                    <div class="ov-stat-label">Memory (MB)</div>
                    <div class="ov-stat-value" style="color:#4ade80;">{profile['memory_usage_mb']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


    with quality_tab:
        total_invalid_dates = sum(scan_report["invalid_dates"].values())
        total_negative_finance = sum(scan_report["negative_finance"].values())
        total_outliers = sum(scan_report["extreme_outliers"].values())
        total_missing = sum(scan_report["missing_values"].values())

        def sev_color(count):
            if count == 0: return "#4ade80", "#1a2e1a", "#2d6a2d"
            elif count < 100: return "#facc15", "#2e2a1a", "#6a5a2d"
            elif count < 1000: return "#fb923c", "#2e1f1a", "#6a3a2d"
            else: return "#f87171", "#3b1f1f", "#7f3f3f"

        c1, c2, c3 = sev_color(total_invalid_dates)
        c5, c6, c7 = sev_color(total_negative_finance)
        c8, c9, c10 = sev_color(total_outliers)
        c11, c12, c13 = sev_color(total_missing)

        st.markdown(f"""
        <style>
        .dq-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 14px;
            margin-bottom: 20px;
        }}
        .dq-card {{
            border-radius: 16px;
            padding: 22px 24px;
            position: relative;
            overflow: hidden;
        }}
        .dq-card-label {{
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .dq-card-value {{
            font-size: 42px;
            font-weight: 900;
            line-height: 1;
            margin-bottom: 6px;
        }}
        .dq-card-tag {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 99px;
            font-size: 11px;
            font-weight: 600;
        }}
        .dq-dot {{
            position: absolute;
            width: 80px; height: 80px;
            border-radius: 50%;
            opacity: 0.1;
            bottom: -20px; right: -20px;
        }}
        .sev-strip {{
            border-radius: 14px;
            padding: 20px 24px;
            background: #111827;
            border: 1px solid #1e293b;
            margin-top: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .sev-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 99px;
            font-size: 13px;
            font-weight: 600;
        }}
        </style>

        <div class="dq-grid">
            <div class="dq-card" style="background:linear-gradient(135deg,{c6},{{"#1e293b"}});border:1px solid {c7};">
                <div class="dq-dot" style="background:{c5};"></div>
                <div class="dq-card-label" style="color:{c5};">Invalid Dates</div>
                <div class="dq-card-value" style="color:{c1};">{total_invalid_dates}</div>
                <div class="dq-card-tag" style="background:{c6};color:{c5};">{get_severity(total_invalid_dates)}</div>
            </div>
            <div class="dq-card" style="background:linear-gradient(135deg,{c6},{{"#1e293b"}});border:1px solid {c7};">
                <div class="dq-dot" style="background:{c5};"></div>
                <div class="dq-card-label" style="color:{c5};">Negative Finance</div>
                <div class="dq-card-value" style="color:{c5};">{total_negative_finance}</div>
                <div class="dq-card-tag" style="background:{c6};color:{c5};">{get_severity(total_negative_finance)}</div>
            </div>
            <div class="dq-card" style="background:linear-gradient(135deg,{c9},{{"#1e293b"}});border:1px solid {c10};">
                <div class="dq-dot" style="background:{c8};"></div>
                <div class="dq-card-label" style="color:{c8};">Outliers</div>
                <div class="dq-card-value" style="color:{c8};">{total_outliers}</div>
                <div class="dq-card-tag" style="background:{c9};color:{c8};">{get_severity(total_outliers)}</div>
            </div>
            <div class="dq-card" style="background:linear-gradient(135deg,{c12},{{"#1e293b"}});border:1px solid {c13};">
                <div class="dq-dot" style="background:{c11};"></div>
                <div class="dq-card-label" style="color:{c11};">Missing Values</div>
                <div class="dq-card-value" style="color:{c11};">{total_missing}</div>
                <div class="dq-card-tag" style="background:{c12};color:{c11};">{get_severity(total_missing)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin:24px 0 16px 0;">
            <div style="font-size:22px;font-weight:800;color:#e2e8f0;letter-spacing:3px;text-transform:uppercase;font-family:'Georgia',serif;">Detailed Data Quality Findings</div>
        </div>
        """, unsafe_allow_html=True)

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

        critical_issues = (total_invalid_dates + total_negative_finance + total_outliers)

        if critical_issues == 0:
            sev_html = '<div class="sev-item" style="background:#1a2e1a;border:1px solid #2d6a2d;color:#4ade80;">✅ No Critical Issues Detected</div>'
        else:
            sev_html = ""
            if total_invalid_dates > 0:
                col, bg, border = sev_color(total_invalid_dates)
                sev_html += f'<div class="sev-item" style="background:{bg};border:1px solid {border};color:{col};">📅 Invalid Dates: {total_invalid_dates} — {get_severity(total_invalid_dates)}</div>'
            if total_negative_finance > 0:
                col, bg, border = sev_color(total_negative_finance)
                sev_html += f'<div class="sev-item" style="background:{bg};border:1px solid {border};color:{col};">💰 Negative Finance: {total_negative_finance} — {get_severity(total_negative_finance)}</div>'
            if total_outliers > 0:
                col, bg, border = sev_color(total_outliers)
                sev_html += f'<div class="sev-item" style="background:{bg};border:1px solid {border};color:{col};">📊 Outliers: {total_outliers} — {get_severity(total_outliers)}</div>'

        st.markdown(f"""
        <div class="sev-strip">
            <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#475569;margin-right:4px;">Issue Severity</div>
            {sev_html}
        </div>
        """, unsafe_allow_html=True)
    

    with ai_tab:
        if health_score >= 85:
            ai_color = "#4ade80"
            ai_bg = "#1a2e1a"
            ai_border = "#2d6a2d"
            ai_status = "High AI Readiness"
            ai_icon = "🟢"
        elif health_score >= 70:
            ai_color = "#facc15"
            ai_bg = "#2e2a1a"
            ai_border = "#6a5a2d"
            ai_status = "Moderate AI Readiness"
            ai_icon = "🟡"
        else:
            ai_color = "#f87171"
            ai_bg = "#3b1f1f"
            ai_border = "#7f3f3f"
            ai_status = "Low AI Readiness"
            ai_icon = "🔴"

        bar_w = health_score

        st.markdown(f"""
        <style>
        .ai-hero {{
            background: linear-gradient(135deg, #0d1f3c, #1a1a3b, #1e293b);
            border: 1px solid #2d3a6a;
            border-radius: 20px;
            padding: 32px 36px;
            position: relative;
            overflow: hidden;
            margin-bottom: 16px;
        }}
        .ai-hero-glow1 {{
            position: absolute;
            width: 250px; height: 250px;
            border-radius: 50%;
            background: #6366f1;
            opacity: 0.07;
            top: -80px; right: -60px;
        }}
        .ai-hero-glow2 {{
            position: absolute;
            width: 150px; height: 150px;
            border-radius: 50%;
            background: #818cf8;
            opacity: 0.05;
            bottom: -50px; left: 100px;
        }}
        .ai-hero-inner {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 24px;
            align-items: center;
        }}
        .ai-title-block {{
            grid-column: span 2;
        }}
        .ai-eyebrow {{
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #6366f1;
            margin-bottom: 8px;
        }}
        .ai-title {{
            font-size: 28px;
            font-weight: 900;
            color: #e2e8f0;
            margin-bottom: 6px;
            font-family: 'Georgia', serif;
            letter-spacing: 1px;
        }}
        .ai-title span {{
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .ai-subtitle {{
            font-size: 13px;
            color: #475569;
        }}
        .ai-score-block {{
            text-align: center;
            background: #111827;
            border: 1px solid #2d3a6a;
            border-radius: 14px;
            padding: 20px;
        }}
        .ai-score-label {{
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #475569;
            margin-bottom: 8px;
        }}
        .ai-score-num {{
            font-size: 42px;
            font-weight: 900;
            color: {ai_color};
            line-height: 1;
            margin-bottom: 10px;
        }}
        .ai-bar-track {{
            background: #1e293b;
            border-radius: 99px;
            height: 6px;
            overflow: hidden;
            margin-bottom: 8px;
        }}
        .ai-bar-fill {{
            height: 6px;
            border-radius: 99px;
            background: {ai_color};
            width: {bar_w}%;
        }}
        .ai-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 99px;
            background: {ai_bg};
            border: 1px solid {ai_border};
            color: {ai_color};
            font-size: 11px;
            font-weight: 600;
        }}
        </style>

        <div class="ai-hero">
            <div class="ai-hero-glow1"></div>
            <div class="ai-hero-glow2"></div>
            <div class="ai-hero-inner">
                <div class="ai-title-block">
                    <div class="ai-eyebrow">Intelligent Analysis Engine</div>
                    <div class="ai-title">Dataset <span>Intelligence</span> Report</div>
                    <div class="ai-subtitle">AI-powered analysis of your dataset's structure, quality, and business readiness</div>
                </div>
                <div class="ai-score-block">
                    <div class="ai-score-label">AI Readiness Score</div>
                    <div class="ai-score-num">{health_score}<span style="font-size:16px;color:#475569;">/100</span></div>
                    <div class="ai-bar-track"><div class="ai-bar-fill"></div></div>
                    <div class="ai-badge">{ai_icon} {ai_status}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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


    with cleaning_tab:
        st.markdown("""
        <style>
        .clean-hero {
            background: linear-gradient(135deg, #1a0f2e, #1e1b3b, #1e293b);
            border: 1px solid #3b2d6a;
            border-radius: 20px;
            padding: 28px 32px;
            position: relative;
            overflow: hidden;
            margin-bottom: 16px;
        }
        .clean-glow1 {
            position: absolute;
            width: 200px; height: 200px;
            border-radius: 50%;
            background: #a855f7;
            opacity: 0.07;
            top: -60px; right: -40px;
        }
        .clean-glow2 {
            position: absolute;
            width: 120px; height: 120px;
            border-radius: 50%;
            background: #7c3aed;
            opacity: 0.05;
            bottom: -40px; left: 150px;
        }
        .clean-eyebrow {
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #a855f7;
            margin-bottom: 8px;
        }
        .clean-title {
            font-size: 26px;
            font-weight: 900;
            font-family: 'Georgia', serif;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        .clean-subtitle {
            font-size: 13px;
            color: #475569;
        }
        .clean-pills {
            display: flex;
            gap: 10px;
            margin-top: 16px;
            flex-wrap: wrap;
        }
        .clean-pill {
            background: #2d1f4a;
            border: 1px solid #5b3a8e;
            border-radius: 99px;
            padding: 5px 14px;
            font-size: 12px;
            color: #c084fc;
            font-weight: 500;
        }
        .handler-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 14px;
            margin-bottom: 16px;
        }
        .handler-card {
            border-radius: 16px;
            padding: 22px 24px;
            position: relative;
            overflow: hidden;
        }
        .handler-dot {
            position: absolute;
            width: 80px; height: 80px;
            border-radius: 50%;
            opacity: 0.1;
            bottom: -20px; right: -20px;
        }
        .handler-icon {
            font-size: 22px;
            margin-bottom: 10px;
        }
        .handler-label {
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .handler-title {
            font-size: 17px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .handler-sub {
            font-size: 12px;
            color: #475569;
        }
        </style>

        <div class="clean-hero">
            <div class="clean-glow1"></div>
            <div class="clean-glow2"></div>
            <div class="clean-eyebrow">Data Preparation Studio</div>
            <div class="clean-title" style="background:linear-gradient(90deg,#c084fc,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                Intelligent Cleaning Engine
            </div>
            <div class="clean-subtitle">Configure how each data issue should be resolved before analysis</div>
            <div class="clean-pills">
                <div class="clean-pill">🗓 Date Repair</div>
                <div class="clean-pill">💰 Finance Correction</div>
                <div class="clean-pill">📊 Outlier Control</div>
                <div class="clean-pill">🏷 Category Normalization</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ Categorical Normalization", expanded=False):
            category_mappings = {}
            if scan_report["inconsistent_categories"]:
                st.info("Review categorical columns below. Similar values may represent the same category.")
                object_cols = list(df.select_dtypes(include="object").columns)
                selected_col = st.selectbox("Select categorical column", object_cols)
                cleaned_series = (df[selected_col].dropna().astype(str).apply(clean_category_text))
                value_counts = (cleaned_series.value_counts().reset_index())
                value_counts.columns = ["Category", "Count"]
                st.dataframe(value_counts, use_container_width=True, height=400)
                category_mappings[selected_col] = {v: v for v in cleaned_series.unique().tolist()}
            else:
                st.success("No inconsistent categorical values detected")

        st.markdown("""
        <div class="handler-grid">
            <div class="handler-card" style="background:linear-gradient(135deg,#1a1f3b,#1e293b);border:1px solid #2d3a7f;">
                <div class="handler-dot" style="background:#818cf8;"></div>
                <div class="handler-icon">🗓</div>
                <div class="handler-label" style="color:#818cf8;">Temporal Integrity</div>
                <div class="handler-title" style="color:#a5b4fc;">Invalid Date Handling</div>
                <div class="handler-sub">Repair or remove malformed date entries</div>
            </div>
            <div class="handler-card" style="background:linear-gradient(135deg,#1a2e1a,#1e293b);border:1px solid #2d6a2d;">
                <div class="handler-dot" style="background:#4ade80;"></div>
                <div class="handler-icon">💰</div>
                <div class="handler-label" style="color:#4ade80;">Financial Integrity</div>
                <div class="handler-title" style="color:#86efac;">Negative Finance Handling</div>
                <div class="handler-sub">Correct impossible negative financial values</div>
            </div>
            <div class="handler-card" style="background:linear-gradient(135deg,#2e1f1a,#1e293b);border:1px solid #6a3a2d;">
                <div class="handler-dot" style="background:#fb923c;"></div>
                <div class="handler-icon">📊</div>
                <div class="handler-label" style="color:#fb923c;">Statistical Integrity</div>
                <div class="handler-title" style="color:#fdba74;">Outlier Handling</div>
                <div class="handler-sub">Cap, transform, or flag extreme values</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🗓 Configure Date Handling", expanded=False):
            date_choices = {}
            date_fill_methods = {}
            if scan_report["invalid_dates"]:
                for col in scan_report["invalid_dates"]:
                    choice = st.selectbox(f"Invalid dates in {col}:",
                        ["drop_rows", "fill_placeholder", "leave_as_nan"], key=f"{col}_main_choice")
                    date_choices[col] = choice
                    if choice == "fill_placeholder":
                        fill_method = st.selectbox(f"Fill strategy for {col}:",
                            ["median_date", "forward_fill", "backward_fill"], key=f"{col}_fill_method")
                        date_fill_methods[col] = fill_method
            else:
                st.success("No invalid dates found.")

        with st.expander("💰 Configure Finance Handling", expanded=False):
            finance_choices = {}
            if scan_report["negative_finance"]:
                for col in scan_report["negative_finance"]:
                    choice = st.selectbox(f"Negative values in {col}:",
                        ["convert_to_absolute", "cap_at_zero", "convert_to_nan", "drop_rows"])
                    finance_choices[col] = choice
            else:
                st.success("No negative finance values found.")

        with st.expander("📊 Configure Outlier Handling", expanded=False):
            outlier_choices = {}
            if scan_report["extreme_outliers"]:
                for col in scan_report["extreme_outliers"]:
                    choice = st.selectbox(f"Outliers in {col}:",
                        ["cap_values_winsorize", "log_transform", "flag_only", "drop_rows"])
                    outlier_choices[col] = choice
            else:
                st.success("No outliers found.")

        user_choices = {
            "Category_Mappings": category_mappings,
            "Invalid_Date_Handling": date_choices,
            "date_fill_methods": date_fill_methods,
            "Handling_Negative_Finance": finance_choices,
            "Handling_Outliers": outlier_choices}

        st.session_state["user_choices"] = user_choices

        if st.button("🧹 Run Intelligent Cleaning"):
            if "user_choices" in st.session_state:
                cleaned_df = clean_dataset_interactive(df, st.session_state["user_choices"])
                st.session_state["cleaned_df"] = cleaned_df


        if "cleaned_df" in st.session_state:
            cleaned_df = st.session_state["cleaned_df"]

            missing_before = df.isnull().sum().sum()
            missing_after = cleaned_df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            completeness_before = round((1 - missing_before / total_cells) * 100, 1)
            completeness_after = round((1 - missing_after / total_cells) * 100, 1)
            rows_delta = df.shape[0] - cleaned_df.shape[0]
            missing_delta = missing_before - missing_after
            missing_arrow = "↓" if missing_delta >= 0 else "↑"
            missing_arrow_color = "#4ade80" if missing_delta >= 0 else "#f87171"
            missing_arrow_bg = "#1a2e1a" if missing_delta >= 0 else "#3b1f1f"
            missing_arrow_border = "#2d6a2d" if missing_delta >= 0 else "#7f3f3f"

            st.markdown(f"""
            <style>
            .clean-result-grid2 {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 14px;
                margin-bottom: 16px;
            }}
            .split-box2 {{
                display: flex;
                align-items: center;
            }}
            .split-side2 {{
                flex: 1;
                text-align: center;
            }}
            .clean-result-card {{
                border-radius: 16px;
                padding: 22px 26px;
                position: relative;
                overflow: hidden;
            }}
            .clean-result-dot {{
                position: absolute;
                width: 80px; height: 80px;
                border-radius: 50%;
                opacity: 0.1;
                bottom: -20px; right: -20px;
            }}
            .clean-result-label {{
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 2.5px;
                text-transform: uppercase;
                margin-bottom: 10px;
            }}
            .clean-result-value {{
                font-size: 38px;
                font-weight: 900;
                line-height: 1;
            }}
            .split-divider {{
                width: 1px;
                height: 60px;
                background: #334155;
                margin: 0 16px;
            }}
            .split-label {{
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 8px;
            }}
            .split-value {{
                font-size: 34px;
                font-weight: 900;
                line-height: 1;
            }}
            .arrow-badge {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 12px;
                border-radius: 99px;
                font-size: 12px;
                font-weight: 700;
                margin-top: 10px;
            }}
            </style>

            <div style="font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#a855f7;margin-bottom:10px;">Cleaning Complete</div>
            <div style="font-size:22px;font-weight:900;font-family:'Georgia',serif;background:linear-gradient(90deg,#c084fc,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:16px;">Cleaned Dataset</div>
            """, unsafe_allow_html=True)

            st.dataframe(cleaned_df.head())

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="clean-result-card" style="background:linear-gradient(135deg,#1a1f3b,#1e293b);border:1px solid #2d3a7f;">
                    <div class="clean-result-dot" style="background:#818cf8;"></div>
                    <div class="clean-result-label" style="color:#818cf8;">Row Count</div>
                    <div class="split-box2">
                        <div class="split-side2">
                            <div class="split-label" style="color:#475569;">Before</div>
                            <div class="split-value" style="color:#a5b4fc;">{df.shape[0]}</div>
                        </div>
                        <div class="split-divider"></div>
                        <div class="split-side2">
                            <div class="split-label" style="color:#475569;">After</div>
                            <div class="split-value" style="color:#e2e8f0;">{cleaned_df.shape[0]}</div>
                        </div>
                    </div>
                    <div class="arrow-badge" style="background:#1a1f3b;border:1px solid #2d3a7f;color:#818cf8;">
                        {"↓ " + str(rows_delta) + " rows removed" if rows_delta > 0 else "✓ No rows removed"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="clean-result-card" style="background:linear-gradient(135deg,#1a2e1a,#1e293b);border:1px solid #2d6a2d;">
                    <div class="clean-result-dot" style="background:#4ade80;"></div>
                    <div class="clean-result-label" style="color:#4ade80;">Missing Values</div>
                    <div class="split-box2">
                        <div class="split-side2">
                            <div class="split-label" style="color:#475569;">Before</div>
                            <div class="split-value" style="color:#f87171;">{missing_before}</div>
                        </div>
                        <div class="split-divider"></div>
                        <div class="split-side2">
                            <div class="split-label" style="color:#475569;">After</div>
                            <div class="split-value" style="color:#86efac;">{missing_after}</div>
                        </div>
                    </div>
                    <div class="arrow-badge" style="background:{missing_arrow_bg};border:1px solid {missing_arrow_border};color:{missing_arrow_color};">
                        {missing_arrow} {abs(missing_delta)} missing values resolved
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                outliers_before = sum(scan_report["extreme_outliers"].values())
                outliers_after = sum(cached_scan(cleaned_df)["extreme_outliers"].values())
                outliers_delta = outliers_before - outliers_after
                st.markdown(f"""
                <div class="clean-result-card" style="background:linear-gradient(135deg,#2e1f1a,#1e293b);border:1px solid #6a3a2d;">
                    <div class="clean-result-dot" style="background:#fb923c;"></div>
                    <div class="clean-result-label" style="color:#fb923c;">Outliers</div>
                    <div class="split-box2">
                        <div class="split-side2">
                            <div class="split-label" style="color:#475569;">Before</div>
                            <div class="split-value" style="color:#f87171;">{outliers_before}</div>
                        </div>
                        <div class="split-divider"></div>
                        <div class="split-side2">
                            <div class="split-label" style="color:#475569;">After</div>
                            <div class="split-value" style="color:#fdba74;">{outliers_after}</div>
                        </div>
                    </div>
                    <div class="arrow-badge" style="background:#2e1f1a;border:1px solid #6a3a2d;color:#fb923c;">
                        {"↓ " + str(outliers_delta) + " outliers resolved" if outliers_delta > 0 else "✓ No outliers changed"}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            working_df = cleaned_df
        else:
            st.info("Configure and run cleaning above to unlock target detection and feature importance.")
            working_df = None

    
    with ml_tab:
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

        