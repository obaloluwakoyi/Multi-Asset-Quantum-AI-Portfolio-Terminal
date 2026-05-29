# app.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Internal Engine Architecture Imports
from pipeline import run_institutional_pipeline, export_portfolio_pdf
from quantum_engine import QISKIT_AVAILABLE

# ─── Page Settings ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Portfolio Terminal", 
    page_icon="⚛️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styles (Dark Institutional Theme) ───────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background-color: #0f172a; }
  section[data-testid="stSidebar"] { background-color: #0d1b2a; border-right: 1px solid #1e3a5f; }
  .terminal-header { 
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); 
      border: 1px solid #00d4ff33; 
      border-radius: 12px; 
      padding: 24px 32px; 
      margin-bottom: 24px; 
  }
  .terminal-header h1 { color: #00d4ff; font-size: 1.8rem; margin: 0; letter-spacing: 0.04em; }
  .terminal-header p { color: #94a3b8; margin: 6px 0 0; font-size: 0.9rem; }
  .kpi-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 18px; text-align: center; }
  .kpi-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .kpi-value { color: #00ffaa; font-size: 1.7rem; font-weight: 700; margin-top: 4px; }
  .section-title { color: #cbd5e1; font-size: 1.1rem; font-weight: 600; margin: 25px 0 12px; border-left: 4px solid #00d4ff; padding-left: 10px; }
  .pipeline-badge { display: inline-block; background: #0f2027; border: 1px solid #0284c7; color: #38bdf8; font-size: 0.7rem; padding: 2px 8px; border-radius: 15px; margin-right: 5px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="#1e293b", 
    plot_bgcolor="#0f172a",
    font=dict(color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROL PANEL
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown('### ⚛️ System Controls')
ticker_input = st.sidebar.text_input("Asset Ticker Registry", value="AAPL, TSLA, MSFT, NVDA, AMZN")
risk_val = st.sidebar.slider("Risk Aversion Level", 1, 10, 4)
q_budget = st.sidebar.number_input("Allocation Budget Units", min_value=2, max_value=30, value=12)
investment_total = st.sidebar.number_input("Total Investment ($)", min_value=1000, value=50000, step=1000)
mc_paths = st.sidebar.select_slider("Monte Carlo Path Iterations", options=[200, 500, 1000, 2000], value=500)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Predictive Modeling Framework")
forecast_mode = st.sidebar.selectbox(
    "Stochastic Prediction Engine",
    ["Pure GBM", "Pure Merton", "Dynamic Blend", "Trigger-Switch (VIX)"]
)

# Render specific sub-parameters based on the selected forecasting engine
blend_ratio = 0.8
vix_value = 18.0

if forecast_mode == "Dynamic Blend":
    blend_ratio = st.sidebar.slider("GBM Weight Ratio", 0.0, 1.0, 0.8, help="Weighting assigned to continuous GBM paths vs Merton Jumps.")
elif forecast_mode == "Trigger-Switch (VIX)":
    vix_value = st.sidebar.number_input("Simulated VIX Market Index", min_value=5.0, max_value=80.0, value=18.0, step=1.0, help="VIX values >= 25 will automatically switch models from GBM to Merton Jump-Diffusion.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Cognitive AI Analyst Settings")
ai_provider = st.sidebar.selectbox("LLM Provider Framework", ["Offline / No-API", "Groq", "Gemini", "OpenRouter", "Ollama"])

model_options = {
    "Offline / No-API": ["Deterministic Summary Engine"],
    "Groq": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "groq/compound", "groq/compound-mini"],
    "Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "OpenRouter": ["meta-llama/llama-3-8b-instruct:free", "meta-llama/llama-3.1-8b-instant:free", "mistralai/mistral-7b-instruct:free"],
    "Ollama": ["llama3", "mistral", "phi3", "llama-3.1-8b-instant"]
}
ai_model = st.sidebar.selectbox("Model Identifier Registry", model_options[ai_provider])
ai_key = st.sidebar.text_input("Secure Endpoint API Key Token", type="password")
ollama_endpoint = st.sidebar.text_input("Ollama Base URI", value="http://localhost:11434/api/generate") if ai_provider == "Ollama" else None

st.sidebar.markdown("---")
run_terminal = st.sidebar.button("🚀 Execute Analytical Workspace Strategy", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE RENDERER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="terminal-header">
  <h1>⚛️ Quantum Portfolio Intelligence Terminal</h1>
  <p>
    <span class="pipeline-badge">1 · Telemetry</span> →
    <span class="pipeline-badge">2 · SamplingVQE Engine</span> →
    <span class="pipeline-badge">3 · Risk Guard</span> →
    <span class="pipeline-badge">4 · Predictive Forecasting Matrix</span>
  </p>
</div>
""", unsafe_allow_html=True)

# If the button hasn't been clicked yet, halt execution smoothly without errors
if not run_terminal:
    st.info("💡 Adjust runtime execution parameters within the left-hand command sidebar drawer and click 'Execute Analytical Workspace Strategy'.")
    st.stop()

# ==============================================================================
# SECURE EXECUTION ZONE (All code below here runs only when run_terminal is True)
# ==============================================================================

# Parse UI states into runtime packets
tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
ai_config_packet = {"provider": ai_provider, "model_name": ai_model, "api_key": ai_key, "endpoint_url": ollama_endpoint}
forecast_config_packet = {"mode": forecast_mode, "blend_ratio": blend_ratio, "vix_value": vix_value}

with st.spinner("📡 Computing structural portfolio strategies and processing optimization matrices..."):
    output = run_institutional_pipeline(
        tickers_list=tickers_list, 
        risk_val=risk_val, 
        q_budget=q_budget, 
        investment_total=investment_total, 
        mc_paths=mc_paths, 
        ai_config=ai_config_packet,
        forecast_config=forecast_config_packet
    )

if not output:
    st.error("The selected engine parameters failed to construct a viable portfolio allocation matrix. Please check inputs.")
    st.stop()

# Parse Core Pipeline Metrics
hist_ann = (output["port_cum"].iloc[-1] - 1) * 100
mc_final = output["simulations"][-1, :]
mc_mean = mc_final.mean()

# ── KPI Dashboard Strip ──
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Hist. Annual Return</div><div class="kpi-value">{hist_ann:+.2f}%</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card"><div class="kpi-label">Horizon Expected Value</div><div class="kpi-value">${mc_mean:,.2f}</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card"><div class="kpi-label">95% Daily Capital VaR</div><div class="kpi-value" style="color:#f87171">${output["compliance"]["var_95"]:,.2f}</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><div class="kpi-label">Optimizer Framework Status</div><div class="kpi-value" style="font-size:1.1rem; color:#38bdf8; padding-top:6px;">{"Qiskit VQE Active" if QISKIT_AVAILABLE else "Classical Fallback"}</div></div>', unsafe_allow_html=True)

# ── AI Executive Summary Block ──
st.markdown('<div class="section-title">🧠 Cognitive AI Quantitative Insight Executive Report</div>', unsafe_allow_html=True)
st.markdown(f'<div style="background:#1e293b; border:1px solid #475569; border-radius:10px; padding:20px; margin-bottom:20px; color:#e2e8f0; line-height:1.6;">{output["ai_narrative"]}</div>', unsafe_allow_html=True)

# ── Charts Allocation Split ──
c_left, c_right = st.columns(2)
with c_left:
    st.markdown('<div class="section-title"> doughnuts Allocation Distribution</div>', unsafe_allow_html=True)
    fig_pie = go.Figure(data=[go.Pie(labels=output["df_report"]["Ticker"], values=output["df_report"]["Weight"], hole=0.45)])
    fig_pie.update_layout(**PLOTLY_TEMPLATE, showlegend=True, height=340, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with c_right:
    st.markdown('<div class="section-title">🎲 Stochastic Forward Volatility Horizon Pathways</div>', unsafe_allow_html=True)
    fig_mc = go.Figure()
    # Display downsampled paths for high performance rendering speed
    for i in range(min(100, mc_paths)):
        fig_mc.add_trace(go.Scatter(y=output["simulations"][:, i], mode="lines", line=dict(width=0.5, color="rgba(0,212,255,0.12)"), showlegend=False))
    fig_mc.add_trace(go.Scatter(y=output["simulations"].mean(axis=1), mode="lines", name="Expected Value", line=dict(color="#00ffaa", width=2)))
    fig_mc.update_layout(**PLOTLY_TEMPLATE, height=340, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_mc, use_container_width=True)

# ── Multi-Year Target Forecasting Framework & Data Export Section ──
st.markdown('<div class="section-title">📊 Multi-Year Portfolio Target Forecasts Matrix</div>', unsafe_allow_html=True)
c_grid, c_pdf = st.columns([3, 1])

with c_grid:
    target_display = output["year_targets"].copy()
    chart_df = target_display.copy()
    fig_targets = go.Figure()
    fig_targets.add_trace(go.Scatter(
        x=chart_df["Target Timeline"],
        y=chart_df["Expected Value ($)"],
        mode="lines+markers",
        name="Expected Value",
        line=dict(color="#22c55e", width=3),
        marker=dict(size=8)
    ))
    fig_targets.add_trace(go.Scatter(
        x=chart_df["Target Timeline"],
        y=chart_df["Upside Bound (P90) ($)"],
        mode="lines+markers",
        name="Upside Bound (P90)",
        line=dict(color="#38bdf8", width=2, dash="dash"),
        marker=dict(size=6)
    ))
    fig_targets.add_trace(go.Scatter(
        x=chart_df["Target Timeline"],
        y=chart_df["Downside Bound (P10) ($)"],
        mode="lines+markers",
        name="Downside Bound (P10)",
        line=dict(color="#ef4444", width=2, dash="dot"),
        marker=dict(size=6)
    ))
    fig_targets.update_layout(
        **PLOTLY_TEMPLATE,
        title="Projected Portfolio Trajectories",
        xaxis_title="Target Timeline",
        yaxis_title="Portfolio Value ($)",
        height=360,
        margin=dict(t=40, b=30, l=40, r=20),
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_targets, use_container_width=True)

    target_display["Expected Value ($)"] = target_display["Expected Value ($)"].map("${:,.2f}".format)
    target_display["Downside Bound (P10) ($)"] = target_display["Downside Bound (P10) ($)"].map("${:,.2f}".format)
    target_display["Upside Bound (P90) ($)"] = target_display["Upside Bound (P90) ($)"].map("${:,.2f}".format)
    st.dataframe(target_display, use_container_width=True, hide_index=True)

with c_pdf:
    st.markdown("<div style='text-align: center; padding-top: 10px;'>", unsafe_allow_html=True)
    try:
        pdf_bytes = export_portfolio_pdf(output, investment_total, forecast_mode)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"Quantum_Portfolio_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.success("PDF Compiled Cleanly.")
    except Exception as e:
        st.error(f"PDF Compilation Fault: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Compliance Risk Audits and Shock Stress Matrices Tabs ──
st.markdown('<div class="section-title">⚖️ Compliance Audit Registers, Stress Profiles, and Exit Plan Targets</div>', unsafe_allow_html=True)
tab_comp, tab_stress, tab_exit = st.tabs(["🔒 Asset Compliance Scans", "💥 Shock Stress Matrix", "🚪 Dynamic Exit Criteria"])

with tab_comp:
    for alert in output["compliance"]["alerts"]:
        if "CRITICAL" in alert: st.error(alert)
        else: st.success(alert)
    st.info(f"**Conditional Value-at-Risk Assessment (95% CVaR Portfolio Tail Drop Target):** ${output['compliance']['cvar_95']:,.2f}")

with tab_stress:
    stress_df = pd.DataFrame.from_dict(output["stress_scenarios"], orient="index")
    stress_df.columns = ["Estimated Value Impact (%)", "Projected Capital Evaluation Value ($)"]
    st.dataframe(stress_df.style.format({"Estimated Value Impact (%)": "{:,.2f}%", "Projected Capital Evaluation Value ($)": "${:,.2f}"}), use_container_width=True)

with tab_exit:
    exit_df = pd.DataFrame.from_dict(output["exit_plans"], orient="index")
    chart_df = exit_df.copy()
    chart_df["Stop-Loss (%)"] = chart_df["Stop-Loss Threshold"].str.replace("%", "", regex=False).astype(float)
    chart_df["Take-Profit (%)"] = chart_df["Take-Profit Target"].str.replace("%", "", regex=False).str.replace("+", "", regex=False).astype(float)

    fig_exit = go.Figure()
    fig_exit.add_trace(go.Bar(
        x=chart_df.index,
        y=chart_df["Take-Profit (%)"],
        name="Take-Profit",
        marker_color="#22c55e"
    ))
    fig_exit.add_trace(go.Bar(
        x=chart_df.index,
        y=chart_df["Stop-Loss (%)"],
        name="Stop-Loss",
        marker_color="#ef4444"
    ))
    fig_exit.update_layout(
        **PLOTLY_TEMPLATE,
        title="Dynamic Exit Criteria: Target vs Risk Threshold",
        xaxis_title="Ticker",
        yaxis_title="Percent",
        height=360,
        margin=dict(t=40, b=30, l=40, r=20),
        barmode="group",
        legend=dict(orientation="h", y=-0.2)
    )

    st.plotly_chart(fig_exit, use_container_width=True)
    st.dataframe(exit_df, use_container_width=True)