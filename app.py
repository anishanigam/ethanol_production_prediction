import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Bioethanol Production Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1B5E20;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #2e7d32;
        box-shadow: 0 4px 6px rgba(0,0,0,0.06);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #388e3c;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load data and model
@st.cache_resource
def load_model_package():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'best_bioethanol_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_clean_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_path = os.path.join(base_dir, 'data', 'cleaned_bioethanol_data.csv')
    if os.path.exists(clean_path):
        return pd.read_csv(clean_path)
    return None

@st.cache_data
def load_raw_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'data', 'raw_bioethanol_data.csv')
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return None

model_package = load_model_package()
df_clean = load_clean_data()
df_raw = load_raw_data()

# Header Section
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<div class="main-title">🌿 Bioethanol Production Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI & Machine Learning-Driven Fermentation Yield Estimation from Heterogeneous Biomass Feedstocks</div>', unsafe_allow_html=True)

with col_head2:
    if model_package:
        st.success(f"**Champion Model**: {model_package.get('model_name', 'Extra Trees')}\n\n**Test R²**: {model_package.get('r2_score', 0.948):.3f}")

# Quick Preset Button Callbacks
preset_data = {
    "Default": {
        "substrate_type": "Lignocellulosic Biomass",
        "substrate_conc": 150.0,
        "pretreatment": "Yes",
        "microorganism": "Wild-type S. cerevisiae",
        "method": "Simultaneous Saccharification & Fermentation",
        "temp": 35.0,
        "ph": 5.0,
        "time": 72.0
    },
    "High-Gravity Molasses": {
        "substrate_type": "Liquid/Sugar-rich Byproduct",
        "substrate_conc": 220.0,
        "pretreatment": "No",
        "microorganism": "Wild-type S. cerevisiae",
        "method": "Batch Fermentation",
        "temp": 30.0,
        "ph": 5.8,
        "time": 68.0
    },
    "Engineered SSF Corncob": {
        "substrate_type": "Lignocellulosic Biomass",
        "substrate_conc": 125.0,
        "pretreatment": "Yes",
        "microorganism": "Genetically Engineered Yeast",
        "method": "Simultaneous Saccharification & Fermentation",
        "temp": 40.0,
        "ph": 5.0,
        "time": 120.0
    },
    "Agro-Industrial Fruit Waste": {
        "substrate_type": "Agro-Industrial Waste",
        "substrate_conc": 100.0,
        "pretreatment": "Yes",
        "microorganism": "Wild-type S. cerevisiae",
        "method": "Batch Fermentation",
        "temp": 30.0,
        "ph": 5.5,
        "time": 48.0
    }
}

# Sidebar Controls
st.sidebar.header("🔬 Fermentation Parameters")
preset_choice = st.sidebar.selectbox("Load Experimental Preset:", list(preset_data.keys()))
active_preset = preset_data[preset_choice]

substrate_types = [
    "Lignocellulosic Biomass", 
    "Liquid/Sugar-rich Byproduct", 
    "Agro-Industrial Waste", 
    "Starch/Fructan Biomass", 
    "Algal Biomass"
]
microorganism_types = [
    "Wild-type S. cerevisiae", 
    "Genetically Engineered Yeast", 
    "Non-Saccharomyces Yeast", 
    "Mixed/Co-culture"
]
fermentation_methods = [
    "Simultaneous Saccharification & Fermentation",
    "Batch Fermentation",
    "Fed-Batch",
    "Repeated-Batch"
]

sub_type = st.sidebar.selectbox(
    "1. Substrate Feedstock Type", 
    substrate_types, 
    index=substrate_types.index(active_preset["substrate_type"])
)

sub_conc = st.sidebar.slider(
    "2. Substrate Concentration (g/L)", 
    min_value=10.0, 
    max_value=350.0, 
    value=float(active_preset["substrate_conc"]), 
    step=5.0,
    help="Equivalent to grams of dry biomass or initial sugars per liter of fermentation broth."
)

pretreat = st.sidebar.radio(
    "3. Biomass Pretreatment Applied?", 
    ["Yes", "No"], 
    index=0 if active_preset["pretreatment"] == "Yes" else 1,
    horizontal=True,
    help="Pretreatments include acid/alkali, steam explosion, or hydrothermal autohydrolysis."
)

microbe = st.sidebar.selectbox(
    "4. Microbial Biocatalyst Type", 
    microorganism_types, 
    index=microorganism_types.index(active_preset["microorganism"])
)

method = st.sidebar.selectbox(
    "5. Fermentation Process Mode", 
    fermentation_methods, 
    index=fermentation_methods.index(active_preset["method"])
)

temp = st.sidebar.slider(
    "6. Fermentation Temperature (°C)", 
    min_value=20.0, 
    max_value=50.0, 
    value=float(active_preset["temp"]), 
    step=0.5
)

ph = st.sidebar.slider(
    "7. Operational Medium pH", 
    min_value=3.5, 
    max_value=7.0, 
    value=float(active_preset["ph"]), 
    step=0.1
)

ferm_time = st.sidebar.slider(
    "8. Fermentation Time (hours)", 
    min_value=12.0, 
    max_value=180.0, 
    value=float(active_preset["time"]), 
    step=4.0
)

# Input Dataframe for Prediction
input_df = pd.DataFrame([{
    'substrate_type': sub_type,
    'substrate_concentration_g_L': sub_conc,
    'pretreatment_used': pretreat,
    'microorganism_type': microbe,
    'fermentation_method': method,
    'fermentation_temperature_C': temp,
    'fermentation_pH': ph,
    'fermentation_time_h': ferm_time
}])

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Live Prediction", 
    "📊 Dataset & EDA Explorer", 
    "🏆 Model Benchmarks", 
    "🧠 Feature Importance", 
    "📁 Batch Prediction"
])

# ================= TAB 1: LIVE PREDICTION =================
with tab1:
    st.subheader("Interactive Fermentation Yield Prediction")
    
    if model_package is not None:
        pipeline = model_package['pipeline']
        pred_val = pipeline.predict(input_df)[0]
        # Physical constraints (cannot be negative)
        pred_val = max(0.0, pred_val)

        # Derived biochemical metrics
        # Gay-Lussac Theoretical Max = 0.511 * substrate concentration (if sugar based)
        theoretical_max = 0.511 * sub_conc
        pct_theoretical = min(100.0, (pred_val / theoretical_max * 100.0)) if theoretical_max > 0 else 0
        volumetric_prod = pred_val / ferm_time if ferm_time > 0 else 0
        ethanol_pct_vv = pred_val / (10.0 * 0.789) # g/L to % v/v

        # Display Key Performance Indicators
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Bioethanol</div>
                <div class="metric-val">{pred_val:.2f} <span style="font-size:1.1rem; color:#2e7d32;">g/L</span></div>
                <div style="color:#555; font-size:0.85rem; margin-top:4px;">≈ {ethanol_pct_vv:.2f}% (v/v)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volumetric Productivity</div>
                <div class="metric-val">{volumetric_prod:.2f} <span style="font-size:1.1rem; color:#2e7d32;">g/L·h</span></div>
                <div style="color:#555; font-size:0.85rem; margin-top:4px;">Rate of ethanol generation</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Theoretical Conversion</div>
                <div class="metric-val">{pct_theoretical:.1f}%</div>
                <div style="color:#555; font-size:0.85rem; margin-top:4px;">Basis: 0.511 g/g Gay-Lussac limit</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            tier_color = "#2E7D32" if pred_val >= 70 else ("#F57F17" if pred_val >= 30 else "#C62828")
            tier_text = "High Commercial" if pred_val >= 70 else ("Pilot / Moderate" if pred_val >= 30 else "Benchtop / Low")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Production Tier</div>
                <div class="metric-val" style="color:{tier_color}; font-size:1.7rem;">{tier_text}</div>
                <div style="color:#555; font-size:0.85rem; margin-top:4px;">Industrial viability assessment</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Biochemical Guidance and Diagnostic Feedback
        col_rec1, col_rec2 = st.columns([1, 1])
        with col_rec1:
            st.markdown("### 🔬 Process Diagnostic Insights")
            if temp < 28 or temp > 42:
                st.warning(f"⚠️ **Temperature Warning**: {temp}°C is outside the typical optimal range (30–38°C) for yeast fermentation. Thermal stress or sluggish metabolism may lower titer.")
            else:
                st.info(f"✅ **Optimal Temperature**: {temp}°C supports high glycolytic enzyme activity in *Saccharomyces cerevisiae*.")
                
            if ph < 4.2 or ph > 6.0:
                st.warning(f"⚠️ **pH Warning**: pH {ph} may inhibit yeast cell viability or elevate bacterial contamination risk.")
            else:
                st.info(f"✅ **Optimal pH**: pH {ph} is within the favorable bioethanol fermentation regime (4.5–5.5).")

            if pretreat == "No" and sub_type == "Lignocellulosic Biomass":
                st.error("⚠️ **No Pretreatment on Lignocellulose**: Intact lignin shields cellulose/hemicellulose, severely limiting fermentable hexose/pentose sugar accessibility.")

        with col_rec2:
            st.markdown("### 📋 Current Simulation Vector")
            st.dataframe(input_df.T.rename(columns={0: "Assigned Value"}), use_container_width=True)

    else:
        st.error("Model file not found. Please train the model using `process_data_and_train.py`.")

# ================= TAB 2: DATASET & EDA =================
with tab2:
    st.subheader("Biochemical Dataset Profiling & Exploratory Analysis")
    view_option = st.radio("Select View:", ["Cleaned & Standardized Dataset (Used for ML)", "Original Raw Literature Dataset"], horizontal=True)
    
    if view_option.startswith("Cleaned") and df_clean is not None:
        st.dataframe(df_clean, use_container_width=True)
        st.write(f"**Dataset Summary:** 58 rows, 9 columns. All units standardized to `g/L`, `°C`, and `hours`.")
        
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df_clean['ethanol_concentration_g_L'], kde=True, color='#2e7d32', ax=ax, bins=12)
            ax.set_title("Ethanol Concentration Distribution (g/L)", fontweight='bold')
            st.pyplot(fig)
            
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='substrate_type', y='ethanol_concentration_g_L', data=df_clean, palette='Set2', ax=ax)
            ax.set_title("Ethanol Titer by Biomass Feedstock", fontweight='bold')
            plt.xticks(rotation=25, ha='right')
            st.pyplot(fig)
            
    elif df_raw is not None:
        st.dataframe(df_raw, use_container_width=True)
        st.write(f"**Raw Dataset Dimensions:** {df_raw.shape[0]} rows, {df_raw.shape[1]} columns with non-standard strings, ranges, and citation links.")

# ================= TAB 3: MODEL BENCHMARKS =================
with tab3:
    st.subheader("Machine Learning Algorithm Comparison")
    st.markdown("""
    We benchmarked seven distinct Machine Learning regression algorithms on a held-out **20% testing split** 
    with **5-Fold Cross-Validation** to find the optimal biocatalytic predictor.
    """)
    
    benchmark_data = [
        {"Model Name": "Extra Trees Regressor", "Algorithm Family": "Ensemble (Randomized Trees)", "R² Score": 0.9480, "MAE (g/L)": 10.23, "RMSE (g/L)": 17.88, "Status": "🏆 Best Model"},
        {"Model Name": "Linear Regression", "Algorithm Family": "Ordinary Least Squares", "R² Score": 0.7770, "MAE (g/L)": 25.66, "RMSE (g/L)": 37.02, "Status": "Parametric Baseline"},
        {"Model Name": "Random Forest", "Algorithm Family": "Ensemble (Bagging)", "R² Score": 0.7040, "MAE (g/L)": 21.66, "RMSE (g/L)": 42.65, "Status": "Tree Ensemble"},
        {"Model Name": "Gradient Boosting", "Algorithm Family": "Ensemble (Boosting)", "R² Score": 0.6916, "MAE (g/L)": 21.19, "RMSE (g/L)": 43.54, "Status": "Gradient Boosting"},
        {"Model Name": "K-Nearest Neighbors", "Algorithm Family": "Instance-based", "R² Score": 0.4731, "MAE (g/L)": 29.42, "RMSE (g/L)": 56.91, "Status": "Non-Parametric"},
        {"Model Name": "Ridge Regression", "Algorithm Family": "Regularized Linear (L2)", "R² Score": 0.3887, "MAE (g/L)": 43.65, "RMSE (g/L)": 61.30, "Status": "Linear Regularized"},
        {"Model Name": "Support Vector Regressor", "Algorithm Family": "Kernel Trick (RBF)", "R² Score": 0.0658, "MAE (g/L)": 36.77, "RMSE (g/L)": 75.78, "Status": "Support Vector"}
    ]
    df_bm = pd.DataFrame(benchmark_data)
    st.dataframe(df_bm, use_container_width=True)
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='R² Score', y='Model Name', data=df_bm, palette='Blues_r', ax=ax)
        ax.set_title("Test Set R² Score Comparison", fontweight='bold')
        ax.set_xlim(0, 1.05)
        st.pyplot(fig)
        
    with col_b2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='MAE (g/L)', y='Model Name', data=df_bm, palette='Reds_r', ax=ax)
        ax.set_title("Test Mean Absolute Error (g/L) - Lower is Better", fontweight='bold')
        st.pyplot(fig)

# ================= TAB 4: FEATURE IMPORTANCE =================
with tab4:
    st.subheader("Feature Importance & Biochemical Attribution")
    st.markdown("""
    Understanding the biochemical parameters that most strongly govern bioethanol production helps bio-process engineers 
    optimize bioreactor operations and maximize biofuel titer.
    """)
    
    if model_package is not None:
        pipe = model_package['pipeline']
        ohe = pipe.named_steps['preprocessor'].named_transformers_['cat']
        cat_cols = model_package['categorical_columns']
        num_cols = model_package['numerical_columns']
        cat_features = list(ohe.get_feature_names_out(cat_cols))
        all_features = num_cols + cat_features
        
        regressor = pipe.named_steps['regressor']
        if hasattr(regressor, 'feature_importances_'):
            imps = regressor.feature_importances_
            df_imp = pd.DataFrame({'Feature': all_features, 'Importance': imps})
            df_imp = df_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(x='Importance', y='Feature', data=df_imp.head(8), palette='viridis', ax=ax)
            ax.set_title("Top 8 Process Features Driving Bioethanol Yield", fontweight='bold', fontsize=12)
            ax.set_xlabel("Gini Feature Importance")
            st.pyplot(fig)
            
            st.markdown("""
            ### 📌 Engineering Insights:
            1. **Substrate Loading ($g/L$)**: Acts as the primary stoichiometric supplier of fermentable hexose/pentose sugars.
            2. **Feedstock Origin (Liquid Sugar byproducts vs. Solid Lignocellulose)**: Direct sugary substrates (molasses, juices) circumvent the recalcitrant enzymatic hydrolysis bottleneck.
            3. **Operational pH & Fermentation Duration**: Directly regulate cell vitality, ethanol tolerance, and completion of yeast batch fermentation.
            """)

# ================= TAB 5: BATCH PREDICTION =================
with tab5:
    st.subheader("Batch Experiment Prediction")
    st.markdown("Upload a CSV file containing multiple fermentation runs to obtain automated bioethanol predictions.")
    
    # Download sample template
    sample_template = pd.DataFrame([
        {
            'substrate_type': 'Lignocellulosic Biomass',
            'substrate_concentration_g_L': 120.0,
            'pretreatment_used': 'Yes',
            'microorganism_type': 'Wild-type S. cerevisiae',
            'fermentation_method': 'Simultaneous Saccharification & Fermentation',
            'fermentation_temperature_C': 37.0,
            'fermentation_pH': 5.0,
            'fermentation_time_h': 96.0
        },
        {
            'substrate_type': 'Liquid/Sugar-rich Byproduct',
            'substrate_concentration_g_L': 200.0,
            'pretreatment_used': 'No',
            'microorganism_type': 'Wild-type S. cerevisiae',
            'fermentation_method': 'Batch Fermentation',
            'fermentation_temperature_C': 30.0,
            'fermentation_pH': 6.0,
            'fermentation_time_h': 30.0
        }
    ])
    
    csv_bytes = sample_template.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Batch Template",
        data=csv_bytes,
        file_name="bioethanol_batch_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV File for Prediction", type=['csv'])
    if uploaded_file is not None and model_package is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            predictions = model_package['pipeline'].predict(batch_df)
            batch_df['Predicted_Ethanol_Concentration_g_L'] = np.clip(predictions, 0, None).round(2)
            batch_df['Predicted_Ethanol_pct_vv'] = (batch_df['Predicted_Ethanol_Concentration_g_L'] / (10 * 0.789)).round(2)
            
            st.success("Batch Prediction Complete!")
            st.dataframe(batch_df, use_container_width=True)
            
            # Download predicted output
            pred_csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=pred_csv,
                file_name="bioethanol_predictions_result.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing uploaded batch file: {e}")

st.markdown("---")

