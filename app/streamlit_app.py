import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📡",
    layout="wide"
)

# ── Custom CSS — dark pro theme ────────────────────────────────────────────────
st.markdown("""
<style>
/* hide streamlit default header/footer */
#MainMenu, footer, header {visibility: hidden;}

/* metric card style */
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2d3149;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #38bdf8; font-size: 2rem; font-weight: 700; }

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #13151f;
    border-right: 1px solid #1e2235;
}

/* tab style */
button[data-baseweb="tab"] {
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    color: #94a3b8;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ef4444;
    border-bottom: 2px solid #ef4444;
}

/* section header */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 24px 0 12px 0;
}
.red-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #ef4444;
    display: inline-block;
    margin-right: 6px;
}

/* upload box */
[data-testid="stFileUploader"] {
    background: #1a1d27;
    border: 1px dashed #3b4263;
    border-radius: 8px;
    padding: 8px;
}

/* divider color */
hr { border-color: #2d3149; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "raw", "churn-bigml-20.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "modellog.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "minmaxscaler.joblib")

PLOTLY_TPL = dict(template="plotly_dark")
PLOTLY_BG  = dict(paper_bgcolor="#13151f", plot_bgcolor="#13151f", font_color="#e2e8f0")


# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_scaler():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

def prepare(df_raw: pd.DataFrame):
    df = df_raw.copy()
    if "State" in df.columns:
        df = df.drop(columns=["State"])
    df["International plan"] = df["International plan"].map({"Yes": 1, "No": 0}).fillna(df["International plan"])
    df["Voice mail plan"]    = df["Voice mail plan"].map({"Yes": 1, "No": 0}).fillna(df["Voice mail plan"])
    df["Churn"] = df["Churn"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(df["Churn"])
    return df

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Churn Dashboard")
    st.caption("Professional Analytics Dashboard")
    st.markdown("---")

    # Upload section
    st.markdown("#### 📂 Upload Dataset (CSV)")
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    st.markdown("<p style='text-align:center; color:#64748b; font-size:0.75rem;'>or</p>", unsafe_allow_html=True)
    use_default = st.button("🗂️  Use Default Dataset", use_container_width=True)  # buttons still use this

    # Choose data source
    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = None
    if "using_default" not in st.session_state:
        st.session_state["using_default"] = False

    if uploaded is not None:
        st.session_state["df_raw"] = pd.read_csv(uploaded)
        st.session_state["using_default"] = False
    elif use_default or st.session_state["using_default"]:
        st.session_state["df_raw"] = pd.read_csv(DATA_PATH)
        st.session_state["using_default"] = True

    df_raw = st.session_state["df_raw"]

    st.markdown("---")

    # Model status
    try:
        load_model_scaler()
        st.success("✅ Model loaded from disk")
    except Exception:
        st.error("❌ Model not found — run notebook first")

    st.markdown("---")
    st.caption("Built with Streamlit · Scikit-learn · Plotly\nLogistic Regression Pipeline")
    st.caption("© 2025 Churn Analytics")

# ──────────────────────────────────────────────────────────────────────────────
# No data loaded yet — show welcome screen
# ──────────────────────────────────────────────────────────────────────────────
if df_raw is None:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
        <h1 style='font-size:2.2rem;'>📡 Churn Prediction Dashboard</h1>
        <p style='color:#94a3b8; font-size:1rem; margin-top:8px;'>
            Intelligent Customer Churn Prediction · Powered by Logistic Regression
        </p>
        <br/>
        <p style='color:#64748b;'>👈  Upload a CSV or click <b>Use Default Dataset</b> in the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Data is loaded — prepare and run model
# ──────────────────────────────────────────────────────────────────────────────
df_clean = prepare(df_raw)
model, scaler = load_model_scaler()

X = df_clean.drop("Churn", axis=1)
y = df_clean["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_scaled = scaler.transform(X_test)
y_proba       = model.predict_proba(X_test_scaled)[:, 1]
y_pred_50     = model.predict(X_test_scaled)

churn_rate = float(y.mean())
accuracy   = accuracy_score(y_test, y_pred_50)
roc_auc    = roc_auc_score(y_test, y_proba)
recall_50  = recall_score(y_test, y_pred_50)

# ── Page title ─────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:2px;'>📡 Churn Prediction Dashboard</h1>
<p style='color:#94a3b8; margin-top:0;'>Intelligent Customer Churn Prediction · Powered by Logistic Regression</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Executive Summary cards ────────────────────────────────────────────────────
st.markdown("<span class='red-dot'></span> **Executive Summary**", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{len(df_raw):,}", help="Full dataset")
c2.metric("Churn Rate",      f"{churn_rate:.1%}", help="Share of churned customers")
c3.metric("Model Accuracy",  f"{accuracy:.1%}",  help="At default threshold 0.5")
c4.metric("ROC-AUC Score",   f"{roc_auc:.3f}",  help="Strong discriminator → closer to 1 is better")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "🔬 Feature Insights",
    "🏆 Model Performance",
    "🧑‍💼 Customer Risk Prediction"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<span class='red-dot'></span> **Dataset Overview**", unsafe_allow_html=True)

    with st.expander("🗄️ Dataset Preview (first 10 rows)"):
        st.dataframe(df_raw.head(10), width='stretch')

    r1, r2, r3 = st.columns(3)
    r1.metric("Rows",    f"{len(df_raw):,}")
    r2.metric("Columns", len(df_raw.columns))
    mem_mb = df_raw.memory_usage(deep=True).sum() / 1e6
    r3.metric("Memory",  f"{mem_mb:.1f} MB")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Missing values bar
    with col_left:
        st.markdown("**Missing Values by Feature**")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        if missing.empty:
            st.success("✅ No missing values found!")
        else:
            fig_miss = px.bar(
                x=missing.values, y=missing.index,
                orientation="h",
                labels={"x": "Missing Count", "y": "Feature"},
                color=missing.values,
                color_continuous_scale="Reds",
                **PLOTLY_TPL
            )
            fig_miss.update_layout(showlegend=False, coloraxis_showscale=False, height=320,
                                   paper_bgcolor="#13151f", plot_bgcolor="#13151f", font_color="#e2e8f0")
            st.plotly_chart(fig_miss, width='stretch')

    # Churn donut
    with col_right:
        st.markdown("**Churn Distribution**")
        churn_vals = df_raw["Churn"].value_counts().reset_index()
        churn_vals.columns = ["Churn", "Count"]
        churn_vals["Label"] = churn_vals["Churn"].map(
            {True: "Churned", False: "Not Churned", 1: "Churned", 0: "Not Churned"}
        ).fillna(churn_vals["Churn"].astype(str))

        fig_pie = px.pie(
            churn_vals, values="Count", names="Label",
            hole=0.5,
            color="Label",
            color_discrete_map={"Churned": "#ef4444", "Not Churned": "#22c55e"},
            **PLOTLY_TPL
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
        fig_pie.update_layout(height=320, showlegend=True,
                              legend=dict(orientation="v", x=1.0, y=0.5),
                              paper_bgcolor="#13151f", plot_bgcolor="#13151f", font_color="#e2e8f0")
        st.plotly_chart(fig_pie, width='stretch')

    st.markdown("---")
    st.markdown("**Numeric Feature Distributions**")

    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
    if "Churn" in numeric_cols:
        numeric_cols.remove("Churn")

    # Grid of histograms
    cols_per_row = 3
    for i in range(0, min(len(numeric_cols), 9), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
            with row_cols[j]:
                color = px.colors.qualitative.Bold[i + j % len(px.colors.qualitative.Bold)]
                fig_h = px.histogram(
                    df_raw, x=col_name,
                    color_discrete_sequence=[color],
                    **PLOTLY_TPL
                )
                fig_h.update_layout(
                    title=col_name, height=220, showlegend=False,
                    paper_bgcolor="#13151f", plot_bgcolor="#13151f",
                    font_color="#e2e8f0",
                    margin=dict(l=10, r=10, t=35, b=20),
                    xaxis_title="", yaxis_title=""
                )
                st.plotly_chart(fig_h, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<span class='red-dot'></span> **Feature Insights**", unsafe_allow_html=True)

    # Top model coefficients
    st.markdown("**Top Feature Coefficients (Logistic Regression)**")
    feature_names = X.columns.tolist()
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs
    }).sort_values("Coefficient")

    coef_df["Direction"] = coef_df["Coefficient"].apply(
        lambda v: "Increases Risk" if v > 0 else "Decreases Risk"
    )

    fig_coef = px.bar(
        coef_df, x="Coefficient", y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map={"Increases Risk": "#ef4444", "Decreases Risk": "#22c55e"},
        labels={"Coefficient": "Log-Odds Coefficient"},
        **PLOTLY_TPL
    )
    fig_coef.update_layout(height=420, legend_title="Impact Direction",
                           paper_bgcolor="#13151f", plot_bgcolor="#13151f", font_color="#e2e8f0")
    st.plotly_chart(fig_coef, width='stretch')
    st.caption("🔴 Red bars → feature pushes model toward **Churn**.  🟢 Green bars → feature reduces churn risk.  Magnitude = strength of influence.")

    st.markdown("---")

    # Correlation heatmap
    st.markdown("**Correlation Heatmap (Numeric Features)**")
    num_df = df_clean.select_dtypes(include="number")
    corr   = num_df.corr().round(2)

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 8},
        showscale=True
    ))
    fig_corr.update_layout(
        title="Pearson Correlation Matrix",
        height=480,
        paper_bgcolor="#13151f", plot_bgcolor="#13151f", font_color="#e2e8f0",
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig_corr, width='stretch')

    st.markdown("---")

    # Box plots of key features by churn
    st.markdown("**Key Numeric Features — Distribution by Churn Status**")
    key_feats = ["Total day minutes", "Customer service calls", "Total intl minutes", "Total eve minutes"]
    key_feats = [f for f in key_feats if f in df_clean.columns]

    box_cols = st.columns(min(len(key_feats), 2))
    for idx, feat in enumerate(key_feats[:4]):
        with box_cols[idx % 2]:
            tmp = df_clean[[feat, "Churn"]].copy()
            tmp["Churn Label"] = tmp["Churn"].map({1: "Churned", 0: "Not Churned"})
            fig_box = px.box(
                tmp, x="Churn Label", y=feat,
                color="Churn Label",
                color_discrete_map={"Churned": "#ef4444", "Not Churned": "#22c55e"},
                **PLOTLY_TPL
            )
            fig_box.update_layout(height=280, showlegend=False,
                                  paper_bgcolor="#13151f", plot_bgcolor="#13151f",
                                  font_color="#e2e8f0",
                                  margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(fig_box, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<span class='red-dot'></span> **Model Performance**", unsafe_allow_html=True)

    # Threshold slider
    st.markdown("**Prediction Threshold**")
    threshold = st.slider(
        "Choose Threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
        help="Lower = catch more churners; Higher = fewer false alarms"
    )
    st.caption(
        f"If the model is ≥ **{int(threshold*100)}%** confident a customer will churn → flagged. "
        "Lowering the threshold catches more churners but also increases false alarms."
    )

    y_pred = (y_proba >= threshold).astype(int)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.1%}")
    m2.metric("Recall",    f"{recall_score(y_test, y_pred):.1%}",    help="Churners we caught")
    m3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.1%}", help="Of flagged, how many truly churned")
    m4.metric("F1 Score",  f"{f1_score(y_test, y_pred, zero_division=0):.1%}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Confusion matrix
    with col_left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 22},
            colorscale=[[0, "#1a1d27"], [1, "#ef4444"]],
            showscale=False
        ))
        fig_cm.update_layout(
            height=360,
            paper_bgcolor="#13151f", plot_bgcolor="#13151f", font_color="#e2e8f0",
            margin=dict(l=10, r=10, t=30, b=20)
        )
        st.plotly_chart(fig_cm, width='stretch')
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"✅ Caught **{tp}** real churners  |  ❌ Missed **{fn}**  |  ⚠️ False alarms: **{fp}**")

    # ROC Curve
    with col_right:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Model (AUC = {roc_auc:.3f})",
            line=dict(color="#38bdf8", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Guess",
            line=dict(color="#475569", width=1.5, dash="dash")
        ))
        fig_roc.update_layout(
            height=360,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.1),
            paper_bgcolor="#13151f", plot_bgcolor="#13151f", font_color="#e2e8f0",
            margin=dict(l=10, r=10, t=30, b=20)
        )
        st.plotly_chart(fig_roc, width='stretch')
        st.caption("Top-left corner = perfect model. Gray diagonal = random guess.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<span class='red-dot'></span> **Predict Single Customer**", unsafe_allow_html=True)
    st.markdown("Fill in the customer's details and we'll predict their churn probability.")
    st.markdown("---")

    X_all    = df_clean.drop("Churn", axis=1)
    med_vals = X_all.median()

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**📞 Usage (minutes)**")
            day_min   = st.number_input("Total Day Minutes",   0.0, 400.0, 180.0, 1.0)
            eve_min   = st.number_input("Total Eve Minutes",   0.0, 400.0, 200.0, 1.0)
            night_min = st.number_input("Total Night Minutes", 0.0, 400.0, 200.0, 1.0)
            intl_min  = st.number_input("Total Intl Minutes",  0.0, 50.0,  10.0,  0.5)

        with col_b:
            st.markdown("**📋 Account Information**")
            cust_svc  = st.number_input("Customer Service Calls", 0, 20, 1)
            intl_plan = st.selectbox("International Plan", ["No", "Yes"])
            vmail     = st.selectbox("Voice Mail Plan", ["No", "Yes"])

        submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

    if submitted:
        row = med_vals.copy()
        row["Total day minutes"]      = day_min
        row["Total eve minutes"]      = eve_min
        row["Total night minutes"]    = night_min
        row["Total intl minutes"]     = intl_min
        row["Customer service calls"] = cust_svc
        row["International plan"]     = 1 if intl_plan == "Yes" else 0
        row["Voice mail plan"]        = 1 if vmail == "Yes" else 0
        row["Total day charge"]       = round(day_min * 0.17, 2)
        row["Total eve charge"]       = round(eve_min * 0.085, 2)
        row["Total night charge"]     = round(night_min * 0.045, 2)
        row["Total intl charge"]      = round(intl_min * 0.27, 2)

        input_df     = pd.DataFrame([row], columns=X_all.columns)
        input_scaled = scaler.transform(input_df)
        prob         = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        res_col, bar_col = st.columns([1, 2])

        with res_col:
            st.metric("Churn Probability", f"{prob:.1%}")
            if prob < 0.30:
                st.success("🟢 **Low Risk**\n\nCustomer is likely to stay.")
            elif prob < 0.60:
                st.warning("🟠 **Medium Risk**\n\nConsider a proactive retention offer.")
            else:
                st.error("🔴 **High Risk**\n\nAct now — offer a discount or reach out directly.")

        with bar_col:
            st.markdown("**Risk Gauge**")
            bar_color = "#ef4444" if prob >= 0.60 else ("#f97316" if prob >= 0.30 else "#22c55e")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                    "bar": {"color": bar_color, "thickness": 0.25},
                    "bgcolor": "#1a1d27",
                    "bordercolor": "#2d3149",
                    "steps": [
                        {"range": [0,  30], "color": "#14532d"},
                        {"range": [30, 60], "color": "#431407"},
                        {"range": [60, 100], "color": "#450a0a"}
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.8,
                        "value": prob * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=240,
                                    paper_bgcolor="#13151f", plot_bgcolor="#13151f", font_color="#e2e8f0",
                                    margin=dict(l=30, r=30, t=20, b=10))
            st.plotly_chart(fig_gauge, width='stretch')
            st.caption("🟢 < 30% Low   |   🟠 30–60% Medium   |   🔴 > 60% High")
