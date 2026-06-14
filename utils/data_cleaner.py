import pandas as pd
import numpy as np

def data_clean(df):
    df = df.drop_duplicates() #1) Remove duplicated rows 
    df = df.dropna(how="all") #2) Remove rows with all cells empty
    df.columns = [col.lower().strip().replace(" ", "_").replace("-", "_") for col in df.columns] #3) Standarize Columns Names
    df = df.replace(r'^\s*$', np.nan, regex=True) #4) Replace Empty Strings with NaN instead
    numeric_cols = df.select_dtypes(include=np.number).columns #5)
    for col in numeric_cols: #5)
        df[col] = df[col].fillna(df[col].median()) #5) Fill Missing Numeric Values
    categorical_cols = df.select_dtypes(include="object").columns #6)
    for col in categorical_cols: #6)
        df[col] = df[col].fillna("Unknown") #6) Fill Missing Categorical Values
    for col in df.columns: #7)
        if "sales" in col.lower() or "revenue" in col.lower(): #7)
            df[col] = (df[col].astype(str).str.replace("$", "").str.replace(",", "")) #7)
            df[col] = pd.to_numeric(df[col], errors="coerce") #7) Remove Curreny/Comma Symbols in Sales Columns
    for col in df.columns: #8)
        if "date" in col: #8)
            df[col] = pd.to_datetime(df[col], errors="coerce") #8) Convert date columns to proper date format
    for col in df.columns: #9)
        if "sales" in col.lower() or "revenue" in col.lower(): #9)
            df = df[df[col] < 0, col] = np.nan #9) Replace Sales columns in negative with Nan
    
    