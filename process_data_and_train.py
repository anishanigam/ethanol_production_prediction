import os
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def parse_target(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ['NR', 'UC', 'nan', '']:
        return np.nan
    # e.g., '21.45 g/L', '20.92 ± 0.16 g/L', '248 g/L (experimental response as reported)'
    m = re.search(r'([\d\.]+)\s*(?:±\s*[\d\.]+)?\s*g/L', s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # e.g. '4.1 g/100 mL' -> 41 g/L
    m2 = re.search(r'([\d\.]+)\s*g/100\s*mL', s, re.IGNORECASE)
    if m2:
        return float(m2.group(1)) * 10.0
    # e.g. '4.4% w/w' -> ~44 g/L
    m3 = re.search(r'([\d\.]+)\s*%\s*w/w', s, re.IGNORECASE)
    if m3:
        return float(m3.group(1)) * 10.0
    # e.g. '15% v/v' -> density of ethanol 0.789 g/mL -> 150 mL/L * 0.789 = 118.35 g/L
    m4 = re.search(r'([\d\.]+)\s*%\s*v/v', s, re.IGNORECASE)
    if m4:
        return float(m4.group(1)) * 10.0 * 0.789
    # e.g. '6.85%' -> ~68.5 g/L
    m5 = re.search(r'([\d\.]+)\s*%', s)
    if m5:
        return float(m5.group(1)) * 10.0
    # Plain numeric
    m6 = re.search(r'^[\D]*([\d\.]+)', s)
    if m6:
        return float(m6.group(1))
    return np.nan

def parse_temp(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ['NR', 'UC', 'nan', '']:
        return np.nan
    m = re.search(r'([\d\.]+)', s)
    return float(m.group(1)) if m else np.nan

def parse_ph(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ['NR', 'UC', 'nan', '']:
        return np.nan
    m = re.search(r'([\d\.]+)', s)
    return float(m.group(1)) if m else np.nan

def parse_time(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ['NR', 'UC', 'nan', '']:
        return np.nan
    if 'day' in s.lower():
        m = re.search(r'([\d\.]+)\s*day', s, re.IGNORECASE)
        if m:
            return float(m.group(1)) * 24.0
    m = re.search(r'([\d\.]+)\s*h', s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r'([\d\.]+)', s)
    return float(m.group(1)) if m else np.nan

def parse_substrate_conc(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ['NR', 'UC', 'nan', '', 'Corncob hydrolysate; concentration NR']:
        return np.nan
    # % w/v, % dry matter, wt.%, °Brix -> 1% = 10 g/L
    m_pct = re.search(r'([\d\.]+)\s*(?:%|wt\.%|°Brix)', s)
    if m_pct:
        return float(m_pct.group(1)) * 10.0
    # ranges like '240 → 300 g/L'
    m_range = re.search(r'([\d\.]+)\s*(?:→|–|-)\s*([\d\.]+)\s*g/L', s)
    if m_range:
        return (float(m_range.group(1)) + float(m_range.group(2))) / 2.0
    # direct g/L
    m_gl = re.search(r'([\d\.]+)\s*g/L', s, re.IGNORECASE)
    if m_gl:
        return float(m_gl.group(1))
    # '4.1 g%'
    m_gp = re.search(r'([\d\.]+)\s*g%', s)
    if m_gp:
        return float(m_gp.group(1)) * 10.0
    return np.nan

def clean_substrate_type(val):
    s = str(val).lower()
    if 'liquid' in s or 'molasses' in s or 'juice' in s:
        return 'Liquid/Sugar-rich Byproduct'
    elif 'lignocellulosic' in s:
        return 'Lignocellulosic Biomass'
    elif 'fruit' in s or 'food' in s or 'waste' in s or 'whey' in s:
        return 'Agro-Industrial Waste'
    elif 'starch' in s or 'fructan' in s:
        return 'Starch/Fructan Biomass'
    elif 'algal' in s:
        return 'Algal Biomass'
    else:
        return 'Lignocellulosic Biomass'

def clean_fermentation_method(val):
    s = str(val).lower()
    if 'repeated' in s:
        return 'Repeated-Batch'
    elif 'ssf' in s or 'saccharification' in s:
        return 'Simultaneous Saccharification & Fermentation'
    elif 'fed-batch' in s:
        return 'Fed-Batch'
    else:
        return 'Batch Fermentation'

def clean_microorganism(row):
    agent = str(row['agent_name']).lower()
    agent_type = str(row['agent_type']).lower()
    if 'recombinant' in agent or 'genetically engineered' in agent_type:
        return 'Genetically Engineered Yeast'
    elif '+' in agent or 'co-culture' in agent_type or 'mixed' in agent_type:
        return 'Mixed/Co-culture'
    elif 'saccharomyces' in agent:
        return 'Wild-type S. cerevisiae'
    else:
        return 'Non-Saccharomyces Yeast'

def clean_pretreatment(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return 'Yes'
    if s == 'no':
        return 'No'
    return 'No'

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'data', 'raw_bioethanol_data.csv')
    clean_path = os.path.join(base_dir, 'data', 'cleaned_bioethanol_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading raw data from: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    print(f"Raw shape: {df_raw.shape}")

    # Standardize Target
    target_series = df_raw['ethanol_concentration'].apply(parse_target)
    valid_mask = target_series.notna()

    df_clean = pd.DataFrame()
    df_valid = df_raw[valid_mask].copy()

    # Standardize and map features
    df_clean['substrate_type'] = df_valid['substrate_type'].apply(clean_substrate_type)
    df_clean['substrate_concentration_g_L'] = df_valid['substrate_concentration'].apply(parse_substrate_conc)
    df_clean['pretreatment_used'] = df_valid['pretreatment_used'].apply(clean_pretreatment)
    df_clean['microorganism_type'] = df_valid.apply(clean_microorganism, axis=1)
    df_clean['fermentation_method'] = df_valid['fermentation_method'].apply(clean_fermentation_method)
    df_clean['fermentation_temperature_C'] = df_valid['fermentation_temperature'].apply(parse_temp)
    df_clean['fermentation_pH'] = df_valid['fermentation_pH'].apply(parse_ph)
    df_clean['fermentation_time_h'] = df_valid['fermentation_time'].apply(parse_time)
    df_clean['ethanol_concentration_g_L'] = target_series[valid_mask]

    # Impute missing values for physical features using biochemical medians
    num_cols = ['substrate_concentration_g_L', 'fermentation_temperature_C', 'fermentation_pH', 'fermentation_time_h']
    for col in num_cols:
        med = df_clean[col].median()
        print(f"Imputing missing in '{col}' with median: {med:.2f}")
        df_clean[col] = df_clean[col].fillna(med)

    # Reorder columns
    ordered_cols = [
        'substrate_type',
        'substrate_concentration_g_L',
        'pretreatment_used',
        'microorganism_type',
        'fermentation_method',
        'fermentation_temperature_C',
        'fermentation_pH',
        'fermentation_time_h',
        'ethanol_concentration_g_L'
    ]
    df_clean = df_clean[ordered_cols].reset_index(drop=True)

    print(f"\nCleaned dataset shape: {df_clean.shape}")
    print(df_clean.describe().round(2))
    df_clean.to_csv(clean_path, index=False)
    print(f"Saved cleaned data to: {clean_path}")

    # Machine Learning Modeling
    X = df_clean.drop(columns=['ethanol_concentration_g_L'])
    y = df_clean['ethanol_concentration_g_L']

    cat_cols = ['substrate_type', 'pretreatment_used', 'microorganism_type', 'fermentation_method']
    num_cols = ['substrate_concentration_g_L', 'fermentation_temperature_C', 'fermentation_pH', 'fermentation_time_h']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0),
        'KNN Regressor': KNeighborsRegressor(n_neighbors=4),
        'Support Vector Regressor': SVR(C=25.0, epsilon=0.2),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
        'Extra Trees Regressor': ExtraTreesRegressor(n_estimators=100, max_depth=6, random_state=42)
    }

    print("\n" + "="*70)
    print("MODEL BENCHMARK RESULTS (Test Set: 20%)")
    print("="*70)
    print(f"{'Model Name':<28} | {'R2 Score':<10} | {'MAE (g/L)':<12} | {'RMSE (g/L)':<12}")
    print("-"*70)

    best_score = -float('inf')
    best_pipe = None
    best_name = None

    for name, model in candidate_models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print(f"{name:<28} | {r2:<10.4f} | {mae:<12.2f} | {rmse:<12.2f}")

        if r2 > best_score:
            best_score = r2
            best_pipe = pipe
            best_name = name

    print("="*70)
    print(f"Top Performing Model: {best_name} with R² = {best_score:.4f}")

    # Retrain best model pipeline on full dataset for maximum generalization in production app
    best_pipe.fit(X, y)

    model_save_path = os.path.join(models_dir, 'best_bioethanol_model.joblib')
    joblib.dump({
        'pipeline': best_pipe,
        'model_name': best_name,
        'feature_names': list(X.columns),
        'categorical_columns': cat_cols,
        'numerical_columns': num_cols,
        'r2_score': best_score
    }, model_save_path)
    print(f"Successfully saved production pipeline to: {model_save_path}")

if __name__ == '__main__':
    main()
