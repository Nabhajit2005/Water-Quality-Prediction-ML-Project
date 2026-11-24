import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="wide")

BASE = Path(__file__).parent

# files saved from your notebook
MODEL_FILES = {
    'Random Forest': 'random_forest_model.pkl',
    'Gradient Boosting': 'gradient_boosting_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
}
RESULTS_FILE = 'model_results.pkl'
PREPROCESSOR_FILE = 'preprocessor.pkl'    

RANGES = {
    'ph':              (0.0, 14.0, 7.0),
    'Hardness':        (45.0, 330.0, 197.0),
    'Solids':          (300.0, 62000.0, 20900.0),
    'Chloramines':     (0.3, 13.2, 7.1),
    'Sulfate':         (120.0, 490.0, 333.0),
    'Conductivity':    (180.0, 760.0, 422.0),
    'Organic_carbon':  (2.0, 28.5, 14.2),
    'Trihalomethanes': (0.7, 125.0, 66.6),
    'Turbidity':       (1.4, 6.8, 4.0),
}
FEATURES = list(RANGES)

def _load(fname):
    path = BASE / fname
    if not path.exists():
        return None
    with open(path, 'rb') as fh:
        return pickle.load(fh)


@st.cache_resource(show_spinner="Loading models...")
def load_all():
    models = {n: m for n, f in MODEL_FILES.items() if (m := _load(f)) is not None}
    return models, _load(RESULTS_FILE), _load(PREPROCESSOR_FILE)


models, results, preprocessor = load_all()

st.title("💧 Water Potability Predictor")

if not models:
    st.error("No model files found. Put these files in the same folder as "
             "`streamlit_app.py`: " + ", ".join(MODEL_FILES.values()))
    st.stop()

missing = [n for n in MODEL_FILES if n not in models]
if missing:
    st.warning(f"Model file not found for: {', '.join(missing)}")


def input_frame(values):
    x = pd.DataFrame([values])[FEATURES]
    if preprocessor is not None:                     # same transform as training
        x = pd.DataFrame(preprocessor.transform(x), columns=FEATURES)
    return x


def proba(model, x):
    cols = list(getattr(model, 'feature_names_in_', FEATURES))
    return float(model.predict_proba(x[cols])[0, 1])


def importances(model):
    cols = list(getattr(model, 'feature_names_in_', FEATURES))
    return pd.Series(model.feature_importances_, index=cols)

st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Choose model", list(models))
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Predict 'potable' when probability >= threshold")
model = models[model_name]
st.caption(f"Active model: **{model_name}**. Educational demo only, "
           "not for real water safety decisions.")

# a Random Forest that never split gives constant output: warn about it
for n, m in models.items():
    try:
        if np.sum(m.feature_importances_) == 0:
            st.warning(f"**{n}** has all-zero feature importances, so its trees "
                       "never split and it predicts the same thing for every input. "
                       "Check settings such as `ccp_alpha`, then retrain and re-save it.")
    except Exception:
        pass

tab_pred, tab_cmp, tab_res, tab_imp = st.tabs(
    ["🔮 Predict", "⚖️ Compare models", "📊 Model results", "🌿 Feature importance"])

with tab_pred:
    st.subheader("Enter water measurements")
    cols = st.columns(3)
    values = {}
    for i, f in enumerate(FEATURES):
        lo, hi, default = RANGES[f]
        step = max((hi - lo) / 200, 0.001)
        values[f] = cols[i % 3].slider(f, lo, hi, default, step)

    x_new = input_frame(values)
    p = proba(model, x_new)

    st.divider()
    left, right = st.columns([2, 1])
    with left:
        if p >= threshold:
            st.success(f"✅ {model_name} predicts: **Potable** ({p:.1%})")
        else:
            st.error(f"❌ {model_name} predicts: **Not potable** "
                     f"(probability of potable {p:.1%})")
        st.progress(min(max(p, 0.0), 1.0))
    with right:
        st.metric("P(potable)", f"{p:.1%}")


with tab_cmp:
    st.subheader("Same input, all models")
    probs = {n: proba(m, x_new) for n, m in models.items()}
    table = pd.DataFrame({
        'Model': list(probs),
        'P(potable)': [f"{v:.1%}" for v in probs.values()],
        'Prediction': ['Potable' if v >= threshold else 'Not potable'
                       for v in probs.values()],
    })
    st.dataframe(table, hide_index=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(list(probs), list(probs.values()),
            color=['seagreen' if n == model_name else 'lightgray' for n in probs])
    ax.axvline(threshold, color='k', ls='--', label=f'threshold {threshold:.2f}')
    ax.set_xlim(0, 1)
    ax.set_xlabel('P(potable)')
    ax.legend()
    st.pyplot(fig)


with tab_res:
    st.subheader("Results saved from your notebook")
    if results is None:
        st.info(f"`{RESULTS_FILE}` not found.")
    else:
        st.dataframe(results)
        numeric = results.select_dtypes('number')
        if not numeric.empty:
            metric = st.selectbox("Plot metric", list(numeric.columns))
            st.bar_chart(numeric[metric])


with tab_imp:
    st.subheader(f"Feature importance: {model_name}")
    imp = importances(model)
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.sort_values().plot.barh(ax=ax, color='seagreen')
    ax.set_xlabel('Importance')
    st.pyplot(fig)