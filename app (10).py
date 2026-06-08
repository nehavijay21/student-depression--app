import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindGuard · Student Depression Screener",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0c10;
    color: #e2e8f0;
}
.stApp { background: #0a0c10; }

/* Header */
.mg-header {
    background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0d1117 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.mg-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.mg-tagline {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 0.25rem;
    font-weight: 300;
    letter-spacing: 0.04em;
}
.badge-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 500;
}
.badge-green { border-color: #065f46; color: #34d399; background: #052e16; }
.badge-blue  { border-color: #1e3a5f; color: #38bdf8; background: #0c1a2e; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}

/* Card panels */
.card {
    background: #0f1623;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 0.8rem;
}

/* Metric pills */
.metric-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
}
.metric-box {
    flex: 1;
    min-width: 110px;
    background: #0f1623;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1;
}
.metric-lbl {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Result banner */
.result-high {
    background: linear-gradient(135deg, #1a0505, #2d0b0b);
    border: 1px solid #7f1d1d;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
.result-low {
    background: linear-gradient(135deg, #021a0a, #052e16);
    border: 1px solid #14532d;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-sub { font-size: 0.88rem; color: #94a3b8; }

/* Risk gauge */
.risk-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Selectbox + slider overrides */
.stSelectbox label, .stSlider label { color: #94a3b8 !important; font-size: 0.82rem !important; }
.stSelectbox > div > div { background: #111827 !important; border-color: #1e293b !important; color: #e2e8f0 !important; }
div[data-baseweb="select"] * { color: #e2e8f0 !important; }
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.03em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Matplotlib dark bg helper */
.stPlotlyChart, .stPyplot { border-radius: 10px; overflow: hidden; }

/* Divider */
hr { border-color: #1e293b !important; }

/* ── Style Streamlit column blocks as cards ── */
div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
    background: #0f1623 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.3rem 1.5rem 1.3rem !important;
}
.col-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #38bdf8;
    padding-bottom: 0.6rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid #1e293b;
    display: block;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader (best = Logistic Regression per notebook evaluation) ──────────
@st.cache_resource(show_spinner="Training models on dataset…")
def load_models():
    df = pd.read_csv('/content/drive/MyDrive/case study/Student Depression Dataset.csv')
    df_clean = df.drop(columns=['id', 'Work Pressure', 'Job Satisfaction'], errors='ignore')
    df_clean['Financial Stress'] = df_clean['Financial Stress'].fillna(df_clean['Financial Stress'].median())
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    for col in ['CGPA', 'Age']:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[(df_clean[col] >= Q1 - 1.5*IQR) & (df_clean[col] <= Q3 + 1.5*IQR)]

    sleep_map = {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3, 'Others': 1}
    diet_map  = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2, 'Others': 1}
    df_clean['Sleep Duration']  = df_clean['Sleep Duration'].map(sleep_map)
    df_clean['Dietary Habits']  = df_clean['Dietary Habits'].map(diet_map)

    le = LabelEncoder()
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    target   = df_clean['Depression']
    features = df_clean.drop(columns=['Depression'])
    scaler   = MinMaxScaler()
    feat_sc  = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        feat_sc, target, test_size=0.2, random_state=42, stratify=target)

    # Train all three key models; pick best (LR per notebook)
    models_dict = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, m in models_dict.items():
        m.fit(X_train, y_train)
        acc  = m.score(X_test, y_test)
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc_val = auc(fpr, tpr)
        cm = confusion_matrix(y_test, m.predict(X_test))
        results[name] = {'model': m, 'acc': acc, 'auc': roc_auc_val, 'fpr': fpr, 'tpr': tpr, 'cm': cm}

    best_name  = max(results, key=lambda k: results[k]['auc'])
    best_model = results[best_name]['model']

    return best_model, best_name, scaler, features.columns.tolist(), results, X_test, y_test

best_model, best_name, scaler, feature_cols, all_results, X_test, y_test = load_models()
best_acc = all_results[best_name]['acc']
best_auc = all_results[best_name]['auc']


# ── HEADER ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="mg-header">
  <div>
    <div class="mg-logo">🧬 MindGuard</div>
    <div class="mg-tagline">Student Mental Health Risk Screener · For Educational Use Only</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;">
    <span class="badge-pill badge-blue">Model: {best_name}</span>
    <span class="badge-pill badge-green">Acc {best_acc*100:.1f}%</span>
    <span class="badge-pill badge-green">AUC {best_auc:.3f}</span>
    <span class="badge-pill">27 k students</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── INPUT FORM ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Student Profile</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.1])

with c1:
    st.markdown('<span class="col-header">👤 Personal</span>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender",     ["Female", "Male"])
    age        = st.selectbox("Age",        list(range(16, 41)), index=5)
    degree     = st.selectbox("Degree",     ["B.Tech/BE", "BSc/BA/BCA", "M.Tech/MSc", "PhD", "Class 12", "Other"])
    profession = st.selectbox("Profession", ["Student", "Working Professional", "Other"])

with c2:
    st.markdown('<span class="col-header">📚 Academic</span>', unsafe_allow_html=True)
    academic_pressure  = st.selectbox("Academic Pressure",   ["1 — Low","2 — Low-Med","3 — Medium","4 — High","5 — Very High"])
    cgpa               = st.selectbox("CGPA Range",          ["Below 5","5 - 6","6 - 7","7 - 8","8 - 9","9 - 10"])
    study_satisfaction = st.selectbox("Study Satisfaction",  ["1 — Very Low","2 — Low","3 — Medium","4 — High","5 — Very High"])
    work_study_hours   = st.selectbox("Work/Study Hours/Day", list(range(1, 17)), index=5)

with c3:
    st.markdown('<span class="col-header">🏥 Health</span>', unsafe_allow_html=True)
    sleep_duration    = st.selectbox("Sleep Duration",                   ["Less than 5 hours","5-6 hours","7-8 hours","More than 8 hours"])
    dietary_habits    = st.selectbox("Dietary Habits",                   ["Unhealthy","Moderate","Healthy"])
    suicidal_thoughts = st.selectbox("Ever had Suicidal Thoughts?",      ["No","Yes"])
    family_history    = st.selectbox("Family History of Mental Illness", ["No","Yes"])

with c4:
    st.markdown('<span class="col-header">💰 Financial</span>', unsafe_allow_html=True)
    financial_stress = st.selectbox("Financial Stress", ["1 — Low","2 — Low-Med","3 — Medium","4 — High","5 — Very High"])
    ap_val = int(academic_pressure[0])
    fs_val = int(financial_stress[0])
    st.markdown(f"""
    <div style='margin-top:1rem;font-size:0.8rem;'>
      <div style='display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid #1e293b;'>
        <span style='color:#64748b;'>Academic Pressure</span>
        <span style='color:{"#ef4444" if ap_val>=4 else "#facc15" if ap_val==3 else "#22c55e"};font-weight:700;'>{ap_val}/5</span>
      </div>
      <div style='display:flex;justify-content:space-between;padding:0.4rem 0;'>
        <span style='color:#64748b;'>Financial Stress</span>
        <span style='color:{"#ef4444" if fs_val>=4 else "#facc15" if fs_val==3 else "#22c55e"};font-weight:700;'>{fs_val}/5</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔬  Run Depression Risk Assessment", use_container_width=True, type="primary")


# ── PREDICTION ──────────────────────────────────────────────────────────────────
if predict_btn:

    sleep_map = {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3}
    diet_map  = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
    cgpa_map  = {'Below 5': 4.5, '5 - 6': 5.5, '6 - 7': 6.5, '7 - 8': 7.5, '8 - 9': 8.5, '9 - 10': 9.5}

    profile = {
        'Gender'                               : 1 if gender == "Male" else 0,
        'Age'                                  : int(age),
        'City'                                 : 0,
        'Profession'                           : ["Student", "Working Professional", "Other"].index(profession),
        'Academic Pressure'                    : int(academic_pressure[0]),
        'CGPA'                                 : cgpa_map[cgpa],
        'Study Satisfaction'                   : int(study_satisfaction[0]),
        'Sleep Duration'                       : sleep_map[sleep_duration],
        'Dietary Habits'                       : diet_map[dietary_habits],
        'Degree'                               : ["B.Tech/BE","BSc/BA/BCA","M.Tech/MSc","PhD","Class 12","Other"].index(degree),
        'Have you ever had suicidal thoughts ?': 1 if suicidal_thoughts == "Yes" else 0,
        'Work/Study Hours'                     : int(work_study_hours),
        'Financial Stress'                     : int(financial_stress[0]),
        'Family History of Mental Illness'     : 1 if family_history == "Yes" else 0,
    }

    input_df     = pd.DataFrame([profile])[feature_cols]
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)
    pred         = best_model.predict(input_scaled)[0]
    proba        = best_model.predict_proba(input_scaled)[0]
    dep_pct      = round(proba[1] * 100, 1)
    no_dep_pct   = round(proba[0] * 100, 1)

    if dep_pct >= 70:
        risk_label = "HIGH RISK"
        risk_color = "#ef4444"
    elif dep_pct >= 40:
        risk_label = "MODERATE RISK"
        risk_color = "#f59e0b"
    else:
        risk_label = "LOW RISK"
        risk_color = "#22c55e"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Assessment Result</div>', unsafe_allow_html=True)

    # Result banner
    banner_class = "result-high" if pred == 1 else "result-low"
    result_emoji = "🔴" if pred == 1 else "🟢"
    result_text  = "Depression Likely Detected" if pred == 1 else "No Depression Detected"
    result_sub   = (
        "This student profile shows indicators consistent with depression. "
        "Please consider professional mental health support."
        if pred == 1 else
        "This student profile does not show strong indicators of depression. "
        "Continue monitoring overall wellbeing."
    )
    st.markdown(f"""
    <div class="{banner_class}">
      <div class="result-title">{result_emoji} {result_text}</div>
      <div class="result-sub">{result_sub}</div>
      <div style="margin-top:1rem;display:flex;gap:1.2rem;flex-wrap:wrap;">
        <div><span style="color:{risk_color};font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;">{dep_pct}%</span>
          <span style="font-size:0.78rem;color:#64748b;margin-left:0.4rem;">Depression probability</span></div>
        <div><span style="color:{risk_color};font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;">{risk_label}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-val" style="color:{risk_color}">{dep_pct}%</div>
        <div class="metric-lbl">Depression Risk</div>
      </div>
      <div class="metric-box">
        <div class="metric-val" style="color:#22c55e">{no_dep_pct}%</div>
        <div class="metric-lbl">No Depression</div>
      </div>
      <div class="metric-box">
        <div class="metric-val" style="color:#38bdf8">{best_acc*100:.1f}%</div>
        <div class="metric-lbl">Model Accuracy</div>
      </div>
      <div class="metric-box">
        <div class="metric-val" style="color:#818cf8">{best_auc:.3f}</div>
        <div class="metric-lbl">ROC-AUC Score</div>
      </div>
      <div class="metric-box">
        <div class="metric-val" style="color:#a78bfa">{best_name.split()[0]}</div>
        <div class="metric-lbl">Best Model</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts section ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:0.5rem">Analysis Charts</div>', unsafe_allow_html=True)

    DARK_BG   = '#0f1623'
    PANEL_BG  = '#111827'
    TEXT_COL  = '#94a3b8'
    GRID_COL  = '#1e293b'
    ACCENT1   = '#38bdf8'
    ACCENT2   = '#818cf8'
    RED_COL   = '#ef4444'
    GREEN_COL = '#22c55e'

    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor':   PANEL_BG,
        'axes.edgecolor':   GRID_COL,
        'axes.labelcolor':  TEXT_COL,
        'xtick.color':      TEXT_COL,
        'ytick.color':      TEXT_COL,
        'text.color':       TEXT_COL,
        'grid.color':       GRID_COL,
        'grid.linewidth':   0.6,
        'font.family':      'DejaVu Sans',
    })

    # ── Row 1: Confidence bar + Model comparison ──────────────────────────────
    ch1, ch2 = st.columns([1, 1.4])

    with ch1:
        fig, ax = plt.subplots(figsize=(5.5, 3))
        fig.patch.set_facecolor(DARK_BG)
        bars = ax.barh(['No Depression', 'Depression'], [no_dep_pct, dep_pct],
                       color=[GREEN_COL, RED_COL], height=0.5,
                       edgecolor='none', alpha=0.9)
        # subtle bg bands
        ax.axvspan(0, 40, alpha=0.04, color=GREEN_COL)
        ax.axvspan(40, 70, alpha=0.04, color='#f59e0b')
        ax.axvspan(70, 115, alpha=0.04, color=RED_COL)
        for bar, val, col in zip(bars, [no_dep_pct, dep_pct], [GREEN_COL, RED_COL]):
            ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                    f'{val}%', va='center', fontsize=12, fontweight='bold', color=col)
        ax.set_xlim(0, 118)
        ax.set_xlabel('Confidence (%)', fontsize=9)
        ax.set_title('Prediction Confidence', fontsize=11, fontweight='bold', color='#e2e8f0', pad=10)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x', alpha=0.4)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig)
        plt.close()

    with ch2:
        model_names = list(all_results.keys())
        accs  = [all_results[n]['acc']*100 for n in model_names]
        aucs_ = [all_results[n]['auc'] for n in model_names]
        x = np.arange(len(model_names))
        w = 0.32
        fig2, ax2 = plt.subplots(figsize=(6.5, 3))
        fig2.patch.set_facecolor(DARK_BG)
        bars_a = ax2.bar(x - w/2, accs, width=w, color=ACCENT1, alpha=0.88, label='Accuracy (%)', edgecolor='none')
        bars_b = ax2.bar(x + w/2, [a*100 for a in aucs_], width=w, color=ACCENT2, alpha=0.88, label='AUC × 100', edgecolor='none')
        for bar in list(bars_a)+list(bars_b):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                     f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7.5, color=TEXT_COL)
        ax2.set_xticks(x)
        ax2.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8.5)
        ax2.set_ylim(75, 98)
        ax2.set_ylabel('Score', fontsize=9)
        ax2.set_title('Model Comparison', fontsize=11, fontweight='bold', color='#e2e8f0', pad=10)
        ax2.legend(fontsize=8, framealpha=0.2, edgecolor=GRID_COL)
        ax2.spines[['top','right']].set_visible(False)
        ax2.grid(axis='y', alpha=0.3)
        # Highlight best
        best_idx = model_names.index(best_name)
        ax2.axvspan(best_idx - 0.45, best_idx + 0.45, alpha=0.06, color=ACCENT2)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig2)
        plt.close()

    # ── Row 2: Feature importance + ROC curve ────────────────────────────────
    ch3, ch4 = st.columns([1.3, 1])

    with ch3:
        # Feature importance / coefficient magnitudes
        if hasattr(best_model, 'feature_importances_'):
            fi = pd.Series(best_model.feature_importances_, index=feature_cols)
        else:
            fi = pd.Series(np.abs(best_model.coef_[0]), index=feature_cols)
        fi = fi.sort_values(ascending=True).tail(10)

        highlight_feats = ['Have you ever had suicidal thoughts ?', 'Academic Pressure', 'Financial Stress', 'Age']
        bar_colors = [RED_COL if f in highlight_feats else ACCENT1 for f in fi.index]

        fig3, ax3 = plt.subplots(figsize=(7, 3.8))
        fig3.patch.set_facecolor(DARK_BG)
        hbars = ax3.barh(fi.index, fi.values, color=bar_colors, alpha=0.88, edgecolor='none', height=0.62)
        for bar in hbars:
            ax3.text(bar.get_width() + fi.values.max()*0.01, bar.get_y()+bar.get_height()/2,
                     f'{bar.get_width():.3f}', va='center', fontsize=7.5, color=TEXT_COL)
        ax3.set_xlabel('Importance / |Coefficient|', fontsize=9)
        ax3.set_title('Top Predictors of Depression', fontsize=11, fontweight='bold', color='#e2e8f0', pad=10)
        ax3.spines[['top','right']].set_visible(False)
        ax3.grid(axis='x', alpha=0.3)
        red_p  = mpatches.Patch(color=RED_COL,   label='Key risk factors', alpha=0.88)
        blue_p = mpatches.Patch(color=ACCENT1, label='Other features',   alpha=0.88)
        ax3.legend(handles=[red_p, blue_p], fontsize=8, framealpha=0.2, edgecolor=GRID_COL)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig3)
        plt.close()

    with ch4:
        # ROC curves for all models
        fig4, ax4 = plt.subplots(figsize=(5, 3.8))
        fig4.patch.set_facecolor(DARK_BG)
        roc_colors = [ACCENT1, ACCENT2, '#fb923c']
        for (name, res), col in zip(all_results.items(), roc_colors):
            lw = 2.2 if name == best_name else 1.2
            ls = '-'  if name == best_name else '--'
            ax4.plot(res['fpr'], res['tpr'], color=col, lw=lw, ls=ls,
                     label=f"{name.split()[0]} (AUC={res['auc']:.3f})")
        ax4.plot([0,1],[0,1], color=GRID_COL, lw=1, ls=':', label='Random')
        ax4.fill_between(all_results[best_name]['fpr'], all_results[best_name]['tpr'], alpha=0.08, color=ACCENT1)
        ax4.set_xlim([-0.02, 1.0])
        ax4.set_ylim([0.0, 1.03])
        ax4.set_xlabel('False Positive Rate', fontsize=9)
        ax4.set_ylabel('True Positive Rate', fontsize=9)
        ax4.set_title('ROC Curves — All Models', fontsize=11, fontweight='bold', color='#e2e8f0', pad=10)
        ax4.legend(fontsize=7.5, framealpha=0.2, edgecolor=GRID_COL)
        ax4.spines[['top','right']].set_visible(False)
        ax4.grid(alpha=0.25)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig4)
        plt.close()

    # ── Row 3: Confusion Matrix + Risk Gauge ──────────────────────────────────
    ch5, ch6 = st.columns([1, 1])

    with ch5:
        cm = all_results[best_name]['cm']
        fig5, ax5 = plt.subplots(figsize=(4.5, 3.5))
        fig5.patch.set_facecolor(DARK_BG)
        cmap = plt.cm.Blues
        im = ax5.imshow(cm, interpolation='nearest', cmap=cmap, alpha=0.85)
        ax5.set_title(f'Confusion Matrix — {best_name}', fontsize=10, fontweight='bold', color='#e2e8f0', pad=10)
        labels = ['No Depression', 'Depression']
        ax5.set_xticks([0,1]); ax5.set_yticks([0,1])
        ax5.set_xticklabels(labels, fontsize=9); ax5.set_yticklabels(labels, fontsize=9)
        ax5.set_xlabel('Predicted', fontsize=9); ax5.set_ylabel('Actual', fontsize=9)
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax5.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                         fontsize=13, fontweight='bold',
                         color='white' if cm[i,j] > thresh else TEXT_COL)
        plt.colorbar(im, ax=ax5, fraction=0.035, pad=0.04)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig5)
        plt.close()

    with ch6:
        # Risk dial using a half-donut
        fig6, ax6 = plt.subplots(figsize=(4.5, 3.5), subplot_kw=dict(aspect='equal'))
        fig6.patch.set_facecolor(DARK_BG)
        ax6.set_facecolor(DARK_BG)

        # Full semicircle segments: Low (0-40), Med (40-70), High (70-100)
        def draw_arc(ax, theta1, theta2, radius, color, alpha=0.9, lw=12):
            import numpy as np
            theta = np.linspace(np.radians(theta1), np.radians(theta2), 100)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle='round')

        # Background arcs (dim)
        draw_arc(ax6, 0, 72,   1.0, GREEN_COL, alpha=0.15)
        draw_arc(ax6, 72, 126, 1.0, '#f59e0b',  alpha=0.15)
        draw_arc(ax6, 126, 180,1.0, RED_COL,   alpha=0.15)

        # Active arc up to dep_pct
        needle_angle = dep_pct * 1.8  # 0-180 degrees
        if dep_pct <= 40:
            draw_arc(ax6, 0, needle_angle, 1.0, GREEN_COL)
        elif dep_pct <= 70:
            draw_arc(ax6, 0, 72, 1.0, GREEN_COL)
            draw_arc(ax6, 72, needle_angle, 1.0, '#f59e0b')
        else:
            draw_arc(ax6, 0, 72, 1.0, GREEN_COL)
            draw_arc(ax6, 72, 126, 1.0, '#f59e0b')
            draw_arc(ax6, 126, needle_angle, 1.0, RED_COL)

        # Needle
        needle_rad = np.radians(needle_angle)
        ax6.annotate('', xy=(0.7*np.cos(needle_rad), 0.7*np.sin(needle_rad)),
                     xytext=(0,0),
                     arrowprops=dict(arrowstyle='->', color='white', lw=2))
        ax6.plot(0, 0, 'o', color='white', markersize=7, zorder=5)

        ax6.text(0, -0.3, f'{dep_pct}%', ha='center', va='center',
                 fontsize=22, fontweight='bold', color=risk_color,
                 fontfamily='DejaVu Sans')
        ax6.text(0, -0.52, risk_label, ha='center', va='center',
                 fontsize=9, color=TEXT_COL, fontweight='bold')

        # Labels
        ax6.text(-0.98, -0.08, 'LOW', ha='center', fontsize=7.5, color=GREEN_COL)
        ax6.text(0, 1.12,    'HIGH', ha='center', fontsize=7.5, color=RED_COL)
        ax6.text(0.98, -0.08, 'LOW', ha='center', fontsize=7.5, color=GREEN_COL)

        ax6.set_xlim(-1.3, 1.3); ax6.set_ylim(-0.75, 1.3)
        ax6.axis('off')
        ax6.set_title('Depression Risk Gauge', fontsize=10, fontweight='bold', color='#e2e8f0', pad=5)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig6)
        plt.close()

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Personalised Insights</div>', unsafe_allow_html=True)

    tips = []
    if int(academic_pressure[0]) >= 4:
        tips.append(("📖 Academic Pressure", "High academic load detected. Consider time-management techniques, Pomodoro study sessions, and speaking with an academic counsellor."))
    if int(financial_stress[0]) >= 4:
        tips.append(("💳 Financial Stress", "Significant financial stress noted. Explore scholarship options, student welfare funds, or campus financial aid offices."))
    if sleep_duration in ["Less than 5 hours"]:
        tips.append(("😴 Sleep Deprivation", "Severely insufficient sleep is a strong depression trigger. Aim for 7–8 hours; set a consistent sleep schedule."))
    if dietary_habits == "Unhealthy":
        tips.append(("🥗 Poor Diet", "Nutritional deficits correlate with mood disorders. Small steps: more vegetables, less junk food, regular meal timings."))
    if suicidal_thoughts == "Yes":
        tips.append(("🆘 Suicidal Ideation", "**Please seek immediate professional support.** Contact iCall (9152987821) or Vandrevala Foundation (1860-2662-345) in India."))
    if family_history == "Yes":
        tips.append(("🧬 Family History", "Genetic predisposition exists. Regular mental health check-ins with a licensed therapist are advisable."))

    if not tips:
        tips.append(("✅ Healthy Profile", "No major risk factors flagged. Maintain your current healthy habits and periodic self-check-ins."))

    tip_cols = st.columns(min(len(tips), 3))
    for i, (title, body) in enumerate(tips):
        with tip_cols[i % len(tip_cols)]:
            st.markdown(f"""
            <div class="card" style="border-left:3px solid #334155;">
              <div style="font-family:'Syne',sans-serif;font-size:0.88rem;font-weight:700;margin-bottom:0.5rem;">{title}</div>
              <div style="font-size:0.82rem;color:#94a3b8;line-height:1.5;">{body}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.75rem;padding:0.5rem 0 1.5rem;">
  ⚠️ MindGuard is for <strong>educational & research purposes only</strong>. 
  It does <em>not</em> constitute a clinical diagnosis. 
  Always consult a qualified mental health professional. · 
  Dataset: Student Depression Survey (27,901 records)
</div>
""", unsafe_allow_html=True)
