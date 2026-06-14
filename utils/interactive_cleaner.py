import pandas as pd
import numpy as np
import difflib
import re

def clean_dataset_interactive(df, user_choices):
    df_clean = df.copy()

    category_mappings = user_choices.get("Category_Mappings", {})
    for col, mapping in category_mappings.items():
        if mapping:
            df_clean = apply_category_mapping(df_clean, col, mapping)
        else:
            df_clean = normalize_categorical_columns(df_clean, [col])

    for col, strategy in user_choices.get("Invalid_Date_Handling", {}).items():
        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce", dayfirst=True)
        if strategy == "drop_rows":
            df_clean = df_clean[df_clean[col].notna()]
        elif strategy == "fill_placeholder":
            fill_method = user_choices.get("date_fill_methods", {}).get(col)

            if fill_method == "median_date":
                median_date = (df_clean[col].dropna().median())
                df_clean[col] = df_clean[col].fillna(median_date)
            elif fill_method == "forward_fill":
                df_clean[col] = df_clean[col].fillna(method="ffill")
            elif fill_method == "backward_fill":
                df_clean[col] = df_clean[col].fillna(method="bfill")
        
        elif strategy == "leave_as_nan":
            pass
    

    for col, strategy in user_choices.get("Handling_Negative_Finance", {}).items():
        if strategy == "convert_to_absolute":
            df_clean[f"{col}_is_refund"] = (df_clean[col] < 0).astype(int)
            df_clean[col] = df_clean[col].abs()
        elif strategy == "cap_at_zero":
            df_clean[col] = df_clean[col].clip(lower=0)
        elif strategy == "convert_to_nan":
            df_clean.loc[df_clean[col] < 0, col] = np.nan
        elif strategy == "drop_rows":
            df_clean = df_clean[df_clean[col] >= 0]
    

    for col, strategy in user_choices.get("Handling_Outliers", {}).items():
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        if strategy == "cap_values_winsorize":
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        elif strategy == "log_transform":
            df_clean[f"{col}_logged"] = np.log1p(df_clean[col])
        elif strategy == "flag_only":
            df_clean[f"{col}_is_outlier"] = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).astype(int)
        elif strategy == "drop_rows":
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def apply_category_mapping(df, col, mapping):
    df_clean = df.copy()
    series = df_clean[col].astype(str).str.strip()
    series = series.replace({"nan": np.nan, "none": np.nan, "": np.nan, "None": np.nan, "NaN": np.nan})
    series = series.str.replace(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", regex=True)
    series = series.str.replace(r"\s*[-–—]+\s*", "-", regex=True)
    series = series.str.replace(r"\s+", " ", regex=True)
    df_clean[col] = series.map(lambda v: mapping.get(v, v) if pd.notna(v) else v)
    return df_clean


def clean_category_text(x):
    if pd.isna(x):
        return np.nan
    
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x

def normalize_categorical_columns(df, columns):
    df_clean = df.copy()
    for col in columns:
        series = df_clean[col].apply(clean_category_text)

        non_null = series.dropna()
        if non_null.empty:
            df_clean[col] = series
            continue

        value_counts = non_null.value_counts()
        unique_vals = value_counts.index.tolist()
        lower_map = {v: v.lower() for v in unique_vals}

        groups = {}
        assigned = {}

        sorted_vals = sorted(unique_vals, key=lambda v: -len(v))

        for val in sorted_vals:
            if val in assigned:
                continue
            val_lower = lower_map[val]
            group_key = val
            for other in unique_vals:
                if other == val or other in assigned:
                    continue
                other_lower = lower_map[other]
                is_prefix = val_lower.startswith(other_lower) or other_lower.startswith(val_lower)
                is_close = difflib.SequenceMatcher(None, val_lower, other_lower).ratio() >= 0.90
                if is_prefix or is_close:
                    groups.setdefault(group_key, [val]).append(other)
                    assigned[other] = group_key
            if val not in assigned:
                groups.setdefault(group_key, [val]).append(val)
                assigned[val] = group_key

        canonical = {}
        for group_key, members in groups.items():
            members = list(set(members))
            best = max(members, key=lambda m: (value_counts.get(m, 0), len(m)))
            best = best.strip().title()
            for m in members:
                canonical[m] = best

        df_clean[col] = series.map(lambda v: canonical.get(v, v) if pd.notna(v) else v)

    return df_clean