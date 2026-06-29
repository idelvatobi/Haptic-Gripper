import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import threading
import time
from datetime import datetime
import io

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Haptic Gripper Dashboard",
    page_icon="🦾",
    layout="wide"
)

MODEL_PATH = "haptic_classifier.joblib"
SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

LABELS = ['rest', 'light_force', 'medium_force', 'full_force']
LABEL_COLORS = {
    'rest':         '#94a3b8',
    'light_force':  '#4ade80',
    'medium_force': '#fbbf24',
    'full_force':   '#f87171',
}

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def extract_features(raw_series: pd.Series) -> np.ndarray:
    s = np.array(raw_series, dtype=np.float64)
    n = len(s)
    rolling_mean = np.array([s[max(0, i-4):i+1].mean() for i in range(n)], dtype=np.float64)
    rolling_std  = np.array([s[max(0, i-4):i+1].std()  for i in range(n)], dtype=np.float64)
    delta        = np.concatenate([[0.0], np.diff(s)])
    normalized   = s / 1023.0
    return np.column_stack([s, rolling_mean, rolling_std, delta, normalized])

def load_csv(source) -> pd.DataFrame:
    if isinstance(source, str):
        with open(source, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    else:
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8-sig')

    sep = ';' if ';' in content.split('\n')[0] else ','
    df = pd.read_csv(io.StringIO(content), sep=sep)
    df.columns = df.columns.str.strip().str.lower()

    # Normalize raw column name
    for col in ['fsr_raw', 'raw']:
        if col in df.columns:
            df['raw'] = df[col]
            break

    return df

def detect_grips(labels: pd.Series, min_duration: int = 3) -> int:
    in_grip, count, dur = False, 0, 0
    for label in labels:
        if label != 'rest':
            dur += 1
            if not in_grip and dur >= min_duration:
                in_grip, count = True, count + 1
        else:
            in_grip, dur = False, 0
    return count

def compute_stability(raw: pd.Series) -> float:
    active = raw[raw > 20]
    if len(active) < 5:
        return 0.0
    return float(round(1.0 - min(active.std() / 300.0, 1.0), 3))

def build_report_context(df: pd.DataFrame, name: str) -> dict:
    duration_s = (df['timestamp_ms'].max() - df['timestamp_ms'].min()) / 1000
    dist = df['predicted_label'].value_counts(normalize=True).mul(100).round(1)
    active = df[df['predicted_label'] != 'rest']['raw']
    spikes = int((df['raw'].diff().abs() > 100).sum())
    return {
        'session_name':      name,
        'duration_s':        round(duration_s, 1),
        'total_samples':     len(df),
        'grip_events':       detect_grips(df['predicted_label']),
        'stability_score':   round(compute_stability(df['raw']) * 100, 1),
        'spike_count':       spikes,
        'force_distribution': {k: float(dist.get(k, 0)) for k in LABELS},
        'active_force': {
            'mean': round(float(active.mean()), 1) if len(active) > 0 else 0,
            'max':  round(float(active.max()),  1) if len(active) > 0 else 0,
            'min':  round(float(active.min()),  1) if len(active) > 0 else 0,
        },
    }

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
_defaults = {
    'model':          None,
    'capture_data':   [],
    'capturing':      False,
    'stop_event':     None,
    'last_analysis':  None,
    'report_context': None,
    'last_report':    None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.model is None and os.path.exists(MODEL_PATH):
    try:
        st.session_state.model = joblib.load(MODEL_PATH)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════
# SERIAL THREAD
# ═══════════════════════════════════════════════════════════════
def _serial_thread(port, baud, data_list, stop_event):
    try:
        conn = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        while not stop_event.is_set():
            try:
                line = conn.readline().decode('utf-8', errors='ignore').strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        data_list.append({
                            'timestamp_ms': int(parts[0]),
                            'raw':          int(parts[1]),
                        })
            except Exception:
                pass
        conn.close()
    except Exception as e:
        data_list.append({'__error__': str(e)})

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🦾 Haptic Gripper")
    st.markdown("*Health 4.0 · Master-Slave MVP*")
    st.divider()

    if st.session_state.model is not None:
        st.success("✅ Classifier loaded")
    else:
        st.warning("⚠️ No classifier — go to Train tab")

    st.divider()
    api_key = st.text_input("🔑 Claude API Key", type="password", placeholder="sk-ant-...")

    if not ANTHROPIC_AVAILABLE:
        st.error("anthropic not installed.\nRun: `pip install anthropic`")

    st.divider()
    with st.expander("ℹ️ About"):
        st.caption(
            "Robotic haptic gripper with FSR force feedback. "
            "Captures Arduino data, classifies grip type with a "
            "Random Forest, and generates AI-powered session reports."
        )
        st.caption("Human Modeling, Processing & Simulation")

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📡  Capture",
    "🤖  Train Classifier",
    "📊  Analyse Session",
    "📝  AI Report",
])

# ───────────────────────────────────────────────────────────────
# TAB 1 · CAPTURE
# ───────────────────────────────────────────────────────────────
with tab1:
    st.header("📡 Live Session Capture")
    st.caption("Connect to Arduino and record FSR data in real time.")

    if not SERIAL_AVAILABLE:
        st.error("pyserial not installed. Run: `pip install pyserial`")
    else:
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            ports = [p.device for p in serial.tools.list_ports.comports()] or ["No ports found"]
            port_sel = st.selectbox("Serial Port", ports)
        with c2:
            baud_sel = st.selectbox("Baud Rate", [9600, 115200])
        with c3:
            session_name = st.text_input("Session name", value=f"session_{datetime.now().strftime('%H%M%S')}")

        cb1, cb2, cb3 = st.columns(3)
        with cb1:
            start_disabled = st.session_state.capturing
            if st.button("▶ Start Capture", disabled=start_disabled,
                         use_container_width=True, type="primary"):
                st.session_state.capture_data = []
                st.session_state.capturing = True
                ev = threading.Event()
                st.session_state.stop_event = ev
                threading.Thread(
                    target=_serial_thread,
                    args=(port_sel, baud_sel, st.session_state.capture_data, ev),
                    daemon=True
                ).start()
                st.rerun()

        with cb2:
            if st.button("⏹ Stop & Save", disabled=not st.session_state.capturing,
                         use_container_width=True):
                if st.session_state.stop_event:
                    st.session_state.stop_event.set()
                st.session_state.capturing = False
                valid = [d for d in st.session_state.capture_data if '__error__' not in d]
                if valid:
                    df_save = pd.DataFrame(valid)
                    path = os.path.join(SESSIONS_DIR, f"{session_name}.csv")
                    df_save.to_csv(path, index=False)
                    st.success(f"✅ Saved {len(df_save)} samples → {path}")
                st.rerun()

        with cb3:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.capture_data = []
                st.rerun()

        valid_data = [d for d in st.session_state.capture_data if '__error__' not in d]
        errors     = [d for d in st.session_state.capture_data if '__error__' in d]

        if errors:
            st.error(f"Serial error: {errors[-1]['__error__']}")

        if st.session_state.capturing:
            st.info(f"🔴 Recording… {len(valid_data)} samples captured")

        if valid_data:
            df_live = pd.DataFrame(valid_data[-300:])

            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(
                x=df_live['timestamp_ms'], y=df_live['raw'],
                mode='lines', name='FSR Raw',
                line=dict(color='#f97316', width=1.5)
            ))
            fig_live.add_hline(y=417, line_dash='dash', line_color='#fbbf24',
                               annotation_text='light → medium')
            fig_live.add_hline(y=600, line_dash='dash', line_color='#f87171',
                               annotation_text='medium → full')
            fig_live.update_layout(
                title="Live FSR Signal (last 300 samples)",
                xaxis_title="Time (ms)", yaxis_title="FSR Raw",
                height=320, margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_live, use_container_width=True)

        if st.session_state.capturing:
            time.sleep(0.5)
            st.rerun()

# ───────────────────────────────────────────────────────────────
# TAB 2 · TRAIN CLASSIFIER
# ───────────────────────────────────────────────────────────────
with tab2:
    st.header("🤖 Train Grip Classifier")
    st.caption("Upload labeled CSV files (must have a `label` column) to train the model.")

    uploaded = st.file_uploader(
        "Upload training CSVs",
        type=['csv'],
        accept_multiple_files=True
    )

    if uploaded:
        dfs = []
        for f in uploaded:
            try:
                df = load_csv(f)
                if 'label' not in df.columns:
                    st.warning(f"⚠️ {f.name}: no `label` column — skipped")
                    continue
                df['source'] = f.name
                dfs.append(df)
                st.success(f"✅ {f.name} — {len(df)} rows, labels: {sorted(df['label'].unique())}")
            except Exception as e:
                st.error(f"Error loading {f.name}: {e}")

        if dfs:
            df_all = pd.concat(dfs, ignore_index=True).dropna(subset=['raw', 'label'])
            df_all['label'] = df_all['label'].str.strip()

            st.subheader("Dataset Overview")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total samples", len(df_all))
            m2.metric("Sessions",      len(dfs))
            m3.metric("Classes",       df_all['label'].nunique())
            m4.metric("Users",         df_all['source'].nunique())

            dist = df_all['label'].value_counts().reset_index()
            dist.columns = ['label', 'count']
            fig_d = px.bar(dist, x='label', y='count', color='label',
                           color_discrete_map=LABEL_COLORS,
                           title="Class Distribution", text_auto=True)
            fig_d.update_layout(height=280, showlegend=False, margin=dict(t=40, b=20))
            st.plotly_chart(fig_d, use_container_width=True)

            if st.button("🚀 Train Classifier", type="primary", use_container_width=True):
                with st.spinner("Training Random Forest (100 trees)…"):
                    X = extract_features(df_all['raw'])
                    y = np.array(df_all['label'].astype(str).tolist())

                    # Filter out labels not in our expected set
                    mask = np.isin(y, LABELS)
                    X, y = X[mask], y[mask]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    clf = RandomForestClassifier(
                        n_estimators=100, max_depth=10,
                        random_state=42, n_jobs=-1
                    )
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    joblib.dump(clf, MODEL_PATH)
                    st.session_state.model = clf

                st.success(f"✅ Model saved! Test accuracy: **{acc * 100:.1f}%**")

                # Confusion matrix
                present_labels = sorted(set(y_test) | set(y_pred))
                cm = confusion_matrix(y_test, y_pred, labels=present_labels)
                fig_cm = px.imshow(
                    cm, x=present_labels, y=present_labels,
                    color_continuous_scale='Blues', text_auto=True,
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="True", color="Count")
                )
                fig_cm.update_layout(height=380)
                st.plotly_chart(fig_cm, use_container_width=True)

                # Feature importance
                feat_names = ['raw', 'rolling_mean', 'rolling_std', 'delta', 'normalized']
                imp_df = pd.DataFrame({
                    'feature':    feat_names,
                    'importance': clf.feature_importances_,
                }).sort_values('importance')
                fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h',
                                 title="Feature Importance",
                                 color='importance', color_continuous_scale='Oranges')
                fig_imp.update_layout(height=250, showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig_imp, use_container_width=True)

# ───────────────────────────────────────────────────────────────
# TAB 3 · ANALYSE SESSION
# ───────────────────────────────────────────────────────────────
with tab3:
    st.header("📊 Analyse Session")

    if st.session_state.model is None:
        st.warning("Train or load a classifier first (🤖 Train Classifier tab).")
    else:
        source = st.radio("Data source", ["Upload CSV", "From captured sessions"], horizontal=True)

        df_s, s_label = None, "session"

        if source == "Upload CSV":
            f = st.file_uploader("Upload session CSV", type=['csv'], key="analyse_upload")
            if f:
                df_s   = load_csv(f)
                s_label = f.name.replace('.csv', '')
        else:
            saved = sorted([x for x in os.listdir(SESSIONS_DIR) if x.endswith('.csv')])
            if saved:
                sel    = st.selectbox("Select session", saved)
                df_s   = load_csv(os.path.join(SESSIONS_DIR, sel))
                s_label = sel.replace('.csv', '')
            else:
                st.info("No captured sessions yet. Use the 📡 Capture tab first.")

        if df_s is not None and 'raw' in df_s.columns:
            with st.spinner("Running classifier…"):
                X_s = extract_features(df_s['raw'])
                df_s['predicted_label'] = st.session_state.model.predict(X_s)

            ctx = build_report_context(df_s, s_label)
            st.session_state.last_analysis  = df_s
            st.session_state.report_context = ctx

            # Metrics row
            st.subheader("Session Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Duration",     f"{ctx['duration_s']} s")
            m2.metric("Samples",      ctx['total_samples'])
            m3.metric("Grip Events",  ctx['grip_events'])
            m4.metric("Stability",    f"{ctx['stability_score']}%")
            m5.metric("Signal Spikes",ctx['spike_count'])

            # Charts
            col_l, col_r = st.columns([3, 1])

            with col_l:
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(
                    x=df_s['timestamp_ms'], y=df_s['raw'],
                    mode='lines', name='FSR Raw',
                    line=dict(color='#64748b', width=1), opacity=0.5
                ))
                for label, color in LABEL_COLORS.items():
                    mask = df_s['predicted_label'] == label
                    if mask.any():
                        fig_t.add_trace(go.Scatter(
                            x=df_s.loc[mask, 'timestamp_ms'],
                            y=df_s.loc[mask, 'raw'],
                            mode='markers', name=label,
                            marker=dict(color=color, size=4),
                        ))
                fig_t.update_layout(
                    title="FSR Signal — Classified Labels",
                    xaxis_title="Time (ms)", yaxis_title="FSR Raw Value",
                    height=360, margin=dict(t=40, b=20)
                )
                st.plotly_chart(fig_t, use_container_width=True)

            with col_r:
                d_pie = df_s['predicted_label'].value_counts()
                fig_pie = px.pie(
                    values=d_pie.values, names=d_pie.index,
                    color=d_pie.index, color_discrete_map=LABEL_COLORS,
                    title="Force Distribution"
                )
                fig_pie.update_layout(height=360, margin=dict(t=40, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            # Active force stats
            st.subheader("Active Force Statistics")
            fa1, fa2, fa3 = st.columns(3)
            fa1.metric("Mean FSR", f"{ctx['active_force']['mean']} / 1023")
            fa2.metric("Max FSR",  f"{ctx['active_force']['max']} / 1023")
            fa3.metric("Min FSR",  f"{ctx['active_force']['min']} / 1023")

            st.info("✅ Analysis complete. Go to 📝 AI Report to generate the full report.")

# ───────────────────────────────────────────────────────────────
# TAB 4 · AI REPORT
# ───────────────────────────────────────────────────────────────
with tab4:
    st.header("📝 AI Session Report")

    if st.session_state.report_context is None:
        st.info("Analyse a session first (📊 Analyse Session tab).")
    else:
        ctx = st.session_state.report_context
        st.subheader(f"Session: `{ctx['session_name']}`")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Summary**")
            st.json({
                'duration':         f"{ctx['duration_s']} s",
                'grip_events':      ctx['grip_events'],
                'stability':        f"{ctx['stability_score']}%",
                'signal_spikes':    ctx['spike_count'],
            })
        with rc2:
            st.markdown("**Force Distribution (%)**")
            st.json(ctx['force_distribution'])

        if not api_key:
            st.warning("Enter your Claude API Key in the sidebar to generate the report.")
        elif not ANTHROPIC_AVAILABLE:
            st.error("Install the anthropic package: `pip install anthropic`")
        else:
            if st.button("🧠 Generate AI Report", type="primary", use_container_width=True):
                prompt = f"""You are an expert analyzing data from a haptic robotic gripper prototype used for surgical simulation training (academic MVP, not a clinical device).

SESSION DATA:
- Session: {ctx['session_name']}
- Duration: {ctx['duration_s']} seconds
- Total samples: {ctx['total_samples']}
- Grip events detected: {ctx['grip_events']}
- Stability score: {ctx['stability_score']}% (100% = perfectly stable signal)
- Sudden signal spikes (>100 units in one step): {ctx['spike_count']}

FORCE DISTRIBUTION (% of session time):
- Rest:         {ctx['force_distribution'].get('rest', 0)}%
- Light force:  {ctx['force_distribution'].get('light_force', 0)}%
- Medium force: {ctx['force_distribution'].get('medium_force', 0)}%
- Full force:   {ctx['force_distribution'].get('full_force', 0)}%

ACTIVE FORCE STATS (when grip was engaged, FSR range 0–1023):
- Mean: {ctx['active_force']['mean']}
- Max:  {ctx['active_force']['max']}
- Min:  {ctx['active_force']['min']}

Write a concise structured report (2-3 sentences per section max):
1. **Session Overview** — brief summary.
2. **Grip Analysis** — quality and force distribution.
3. **Stability & Signal Quality** — stability score and spikes.
4. **Key Observations** — 2 bullet points max.
5. **Recommendations** — 2 bullet points max.
6. **Overall Assessment** — one short paragraph.

Be concise. Professional tone. Academic MVP, not a clinical device."""

                with st.spinner("Generating report with Claude…"):
                    try:
                        client = anthropic.Anthropic(api_key=api_key)
                        resp = client.messages.create(
                            model="claude-sonnet-4-5",
                            max_tokens=1000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.session_state.last_report = resp.content[0].text
                    except Exception as e:
                        st.error(f"API error: {e}")

        if st.session_state.last_report:
            st.divider()
            st.markdown(st.session_state.last_report)
            st.divider()
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=st.session_state.last_report,
                file_name=f"report_{ctx['session_name']}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
