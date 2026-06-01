import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #8B0000, #C0392B, #E74C3C);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p  { font-size: 1.1rem; margin: 0.5rem 0 0; opacity: 0.9; }

    .result-good {
        background: linear-gradient(135deg, #1a6b3c, #27AE60);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.8rem; font-weight: bold;
        box-shadow: 0 4px 15px rgba(39,174,96,0.4);
    }
    .result-bad {
        background: linear-gradient(135deg, #7b1a1a, #C0392B);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.8rem; font-weight: bold;
        box-shadow: 0 4px 15px rgba(192,57,43,0.4);
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        text-align: center; border-left: 4px solid #8B0000;
    }
    .stSlider > div > div { color: #8B0000; }
    .section-title {
        font-size: 1.3rem; font-weight: bold; color: #8B0000;
        border-bottom: 2px solid #8B0000; padding-bottom: 0.3rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'wine_quality_knn_model.pkl'
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Run the Jupyter notebook first to generate 'wine_quality_knn_model.pkl'")
        st.stop()
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model_data = load_model()
model   = model_data['model']
scaler  = model_data['scaler']
features = model_data['feature_names']
best_k  = model_data['best_k']
test_acc = model_data['test_accuracy']

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🍷 Wine Quality Predictor</h1>
    <p>KNN Classifier · Red Wine Physicochemical Analysis · ML Project</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Wine Properties")
    st.markdown("Adjust sliders to match your wine sample:")
    st.markdown("---")

    fixed_acidity     = st.slider("Fixed Acidity (g/dm³)",     4.6, 15.9, 8.3,  0.1,
                                   help="Tartaric acid content. Typical: 7–9")
    volatile_acidity  = st.slider("Volatile Acidity (g/dm³)",  0.12, 1.58, 0.52, 0.01,
                                   help="Acetic acid — too high = vinegar taste")
    citric_acid       = st.slider("Citric Acid (g/dm³)",        0.0,  1.0,  0.27, 0.01,
                                   help="Adds freshness. 0 = none")
    residual_sugar    = st.slider("Residual Sugar (g/dm³)",     1.2, 15.5, 2.5,  0.1,
                                   help="Sugar after fermentation")
    chlorides         = st.slider("Chlorides (g/dm³)",          0.012, 0.611, 0.087, 0.001,
                                   help="Salt content in wine")
    free_sulfur_diox  = st.slider("Free Sulfur Dioxide (mg/dm³)", 1, 72,  16,
                                   help="Free SO₂ — antimicrobial agent")
    total_sulfur_diox = st.slider("Total Sulfur Dioxide (mg/dm³)", 6, 289, 46,
                                   help="Bound + Free SO₂")
    density           = st.slider("Density (g/cm³)",             0.990, 1.004, 0.9967, 0.0001,
                                   help="Density relative to water")
    pH                = st.slider("pH",                           2.74, 4.01, 3.31, 0.01,
                                   help="0 = very acidic, 14 = very basic")
    sulphates         = st.slider("Sulphates (g/dm³)",            0.33, 2.0,  0.66, 0.01,
                                   help="Potassium sulphate additive")
    alcohol           = st.slider("Alcohol (%)",                  8.4, 14.9, 10.4, 0.1,
                                   help="Alcohol by volume percentage")

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Quality", use_container_width=True, type="primary")

# ─── Main Layout ────────────────────────────────────────────────────────────
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown('<div class="metric-card"><b>Algorithm</b><br>K-Nearest Neighbors</div>',
                unsafe_allow_html=True)
with col_info2:
    st.markdown(f'<div class="metric-card"><b>Best K</b><br>K = {best_k}</div>',
                unsafe_allow_html=True)
with col_info3:
    st.markdown(f'<div class="metric-card"><b>Model Accuracy</b><br>{test_acc}%</div>',
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Prediction ─────────────────────────────────────────────────────────────
if predict_btn:
    input_values = [
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_diox, total_sulfur_diox, density,
        pH, sulphates, alcohol
    ]
    input_df = pd.DataFrame([input_values], columns=features)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    class_labels = model.classes_

    prob_bad  = probabilities[list(class_labels).index('Bad')]  if 'Bad'  in class_labels else 0
    prob_good = probabilities[list(class_labels).index('Good')] if 'Good' in class_labels else 0

    # Result display
    col_res, col_proba = st.columns([1, 1])

    with col_res:
        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
        if prediction == 'Good':
            st.markdown(
                '<div class="result-good">✅ Good Quality Wine<br>'
                '<span style="font-size:0.9rem;opacity:0.9">Quality Score ≥ 6</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-bad">❌ Bad Quality Wine<br>'
                '<span style="font-size:0.9rem;opacity:0.9">Quality Score < 6</span></div>',
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence bar
        st.markdown("**Confidence Scores:**")
        st.progress(prob_good, text=f"Good Wine: {prob_good*100:.1f}%")
        st.progress(prob_bad,  text=f"Bad Wine:  {prob_bad*100:.1f}%")

    with col_proba:
        st.markdown('<div class="section-title">📊 Probability Chart</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(['Bad Quality', 'Good Quality'], [prob_bad, prob_good],
                      color=['#C0392B', '#27AE60'], width=0.5, edgecolor='white', linewidth=2)
        for bar, val in zip(bars, [prob_bad, prob_good]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val*100:.1f}%', ha='center', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_title(f'KNN Prediction (K={best_k})', fontsize=12, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Input summary table
    st.markdown("---")
    st.markdown('<div class="section-title">📋 Input Feature Summary</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        'Feature': features,
        'Your Input': input_values,
        'Dataset Mean': [8.3, 0.53, 0.27, 2.5, 0.087, 15.9, 46.5, 0.9967, 3.31, 0.658, 10.4],
        'Status': ['↑ High' if v > m else ('↓ Low' if v < m else '→ Normal')
                   for v, m in zip(input_values,
                   [8.3, 0.53, 0.27, 2.5, 0.087, 15.9, 46.5, 0.9967, 3.31, 0.658, 10.4])]
    })
    st.dataframe(summary_df.set_index('Feature').style.applymap(
        lambda v: 'color: green' if '↑' in str(v) else ('color: red' if '↓' in str(v) else ''),
        subset=['Status']
    ), use_container_width=True)

# ─── About Section ────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About This Project"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **🍷 Wine Quality Prediction — KNN Classifier**
        
        This ML project predicts whether a red wine is **Good** or **Bad** quality
        based on 11 physicochemical properties measured in a laboratory.
        
        **Dataset:** Red Wine Quality (UCI ML Repository)
        - 1,599 wine samples
        - 11 input features
        - Quality scores 3–8
        
        **Algorithm:** K-Nearest Neighbors (KNN)
        - Distance-based classification
        - Requires feature scaling (StandardScaler)
        - Optimal K selected via 10-fold cross-validation
        """)
    with col_b:
        st.markdown("""
        **📊 Key Features:**
        
        | Feature | Impact on Quality |
        |---------|------------------|
        | Alcohol | High = Better |
        | Volatile Acidity | Low = Better |
        | Sulphates | High = Better |
        | Citric Acid | High = Better |
        | Chlorides | Low = Better |
        
        **Target:** `Good` (quality ≥ 6) or `Bad` (quality < 6)
        
        **Model Performance:** ~82% accuracy on test set
        """)

st.markdown("""
<div style="text-align:center; color: #888; margin-top: 2rem; font-size: 0.85rem;">
    🍷 Wine Quality Predictor · KNN Classifier · Built with Streamlit & scikit-learn
</div>
""", unsafe_allow_html=True)
