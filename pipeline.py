# pipeline.py
import pandas as pd
from fpdf import FPDF
from datetime import datetime

# Internal Engine Architecture Core Module Modules
from quantum_engine import fetch_market_data, run_quantum_optimization
from compliance_engine import run_compliance_audit, execute_stress_scenarios, generate_dynamic_exit_plan
from predict import run_adaptive_forecasting_engine, compile_year_target_table
from ai_analyst import get_ai_narrative

def run_institutional_pipeline(tickers_list, risk_val, q_budget, investment_total, mc_paths, ai_config, forecast_config):
    # 1. Market Data Telemetry Feed
    mu, sigma, returns, price_data = fetch_market_data(tuple(tickers_list))
    valid_tickers = list(returns.columns)
    
    # 2. Optimization Configuration Core
    selection = run_quantum_optimization(mu, sigma, valid_tickers, risk_val, q_budget)
    
    # 3. Process Position Table Map
    portfolio_report = []
    total_units = sum(selection) if sum(selection) > 0 else 1
    for i, units in enumerate(selection):
        if units > 0 and i < len(valid_tickers):
            weight = units / total_units
            portfolio_report.append({
                "Ticker": valid_tickers[i],
                "Units": int(units),
                "Weight": weight,
                "Amount_USD": investment_total * weight,
                "Exp_Ann_Return": mu[i] * 252 * 100
            })
    df_report = pd.DataFrame(portfolio_report)
    
    if df_report.empty:
        return None
        
    # 4. Generate Core Portfolio Series
    weights_series = pd.Series(df_report.set_index("Ticker")["Weight"].to_dict())
    port_daily = returns[df_report["Ticker"]].mul(weights_series, axis=1).sum(axis=1)
    port_cum = (1 + port_daily).cumprod()
    
    # 5. Advanced Stochastic Predictive Engines (GBM + Merton Integration)
    simulations = run_adaptive_forecasting_engine(
        port_daily=port_daily,
        initial_capital=investment_total,
        paths=mc_paths,
        mode=forecast_config["mode"],
        blend_ratio=forecast_config["blend_ratio"],
        vix_value=forecast_config["vix_value"]
    )
    
    # Compute Year Targets Grid
    df_year_targets = compile_year_target_table(simulations, investment_total)
    
    # 6. Compliance, Risk Audits & Shock Analysis Frameworks
    audit_results = run_compliance_audit(df_report, returns, investment_total)
    stress_results = execute_stress_scenarios(df_report, returns, investment_total)
    exit_plans = generate_dynamic_exit_plan(df_report, returns)

    def _extract_stress_value(value):
        if isinstance(value, dict):
            return value.get("impact_pct", value.get("Estimated Value Impact (%)", 0.0))
        if hasattr(value, "iloc"):
            return float(value.iloc[0]) if len(value) else 0.0
        if isinstance(value, (list, tuple)):
            return float(value[0]) if value else 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    # 7. Enhanced AI Executive Insight Summary Generation Payload
    portfolio_payload = {
        "allocations": df_report[["Ticker", "Units", "Weight", "Amount_USD", "Exp_Ann_Return"]].to_dict(orient="records"),
        "hist_ann": (port_cum.iloc[-1] - 1) * 100,
        "var_95": audit_results["var_95"],
        "cvar_95": audit_results["cvar_95"],
        "compliance_alerts": audit_results["alerts"],
        "stress_test_losses": {
            k: _extract_stress_value(v) for k, v in stress_results.items()
        },
        "exit_matrix": exit_plans,
        "forecasting_parameters": {
            "engine_mode": forecast_config["mode"],
            "vix_simulated": forecast_config["vix_value"],
            "expected_5yr_value": simulations[-1, :].mean()
        }
    }

    ai_narrative = get_ai_narrative(
        provider=ai_config["provider"],
        model_name=ai_config["model_name"],
        api_key=ai_config["api_key"],
        endpoint_url=ai_config["endpoint_url"],
        portfolio_data=portfolio_payload
    )
    
    return {
        "df_report": df_report,
        "returns": returns,
        "port_cum": port_cum,
        "simulations": simulations,
        "year_targets": df_year_targets,
        "compliance": audit_results,
        "stress_scenarios": stress_results,
        "exit_plans": exit_plans,
        "ai_narrative": ai_narrative
    }

def export_portfolio_pdf(output, investment_total, forecast_mode):
    """Compiles structured quantitative terminal output and AI trade tickets into a PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)

    # Document Header Title Block
    pdf.set_text_color(15, 32, 39)
    pdf.cell(0, 12, "Quantum Portfolio Terminal - Executive Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated on Adaptive Quantum-AI Engine: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(8)

    # 1. Core Strategy Underpinnings Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Executive Strategy Summary Parameters", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"- Total Invested Capital Base Portfolio: ${investment_total:,.2f}", ln=True)
    pdf.cell(0, 6, f"- Downstream Predictive Engine Mode: {forecast_mode}", ln=True)
    pdf.cell(0, 6, f"- Tail Exposure VaR Threshold Target: ${output['compliance']['var_95']:,.2f}", ln=True)
    pdf.ln(5)

    # 2. Capital Allocation Table Structure
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Optimized Position Weights Framework Allocation", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 6, "Ticker", border=1)
    pdf.cell(40, 6, "Units", border=1)
    pdf.cell(50, 6, "Allocation Weight", border=1)
    pdf.cell(50, 6, "Value Assigned ($)", border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for _, row in output["df_report"].iterrows():
        pdf.cell(40, 6, str(row['Ticker']), border=1)
        pdf.cell(40, 6, str(int(row['Units'])), border=1)
        pdf.cell(50, 6, f"{row['Weight']:.2%}", border=1)
        pdf.cell(50, 6, f"${row['Amount_USD']:,.2f}", border=1)
        pdf.ln()
    pdf.ln(5)

    # 3. Multi-Year Targeting Grid
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Multi-Year Predictive Target Trajectories", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(45, 6, "Target Timeline", border=1)
    pdf.cell(45, 6, "Expected Value ($)", border=1)
    pdf.cell(45, 6, "Downside (P10) ($)", border=1)
    pdf.cell(45, 6, "CAGR Target", border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for _, row in output["year_targets"].iterrows():
        pdf.cell(45, 6, str(row['Target Timeline']), border=1)
        pdf.cell(45, 6, f"${row['Expected Value ($)']:,.2f}", border=1)
        pdf.cell(45, 6, f"${row['Downside Bound (P10) ($)']:,.2f}", border=1)
        pdf.cell(45, 6, str(row['Implied Return Engine (CAGR)']), border=1)
        pdf.ln()
    pdf.ln(5)

    # 4. Cognitive AI Insight & Rebalancing Tickets Section
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. Cognitive AI Strategic Rebalancing Tickets & Evaluation", ln=True)
    pdf.set_font("Helvetica", "", 10)

    def _sanitize_pdf_text(value):
        if value is None:
            return "N/A"
        if not isinstance(value, str):
            value = str(value)
        text = value.replace("\r", " ").replace("\n", " ")
        return text.encode("latin-1", errors="ignore").decode("latin-1")

    ai_text = _sanitize_pdf_text(output.get("ai_narrative", "No executive narrative available."))
    for line in ai_text.split("\n"):
        clean_line = _sanitize_pdf_text(line.strip())
        if clean_line:
            pdf.multi_cell(0, 6, clean_line)
    pdf.ln(5)

    # 5. Exit Rules Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "5. Dynamic Exit Matrix Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    exit_matrix = output.get("exit_plans", {})
    for ticker, rules in exit_matrix.items():
        stop_loss = _sanitize_pdf_text(rules.get("Stop-Loss Threshold", rules.get("Trailing Stop-Loss", "N/A")))
        take_profit = _sanitize_pdf_text(rules.get("Take-Profit Target", "N/A"))
        pdf.multi_cell(0, 6, _sanitize_pdf_text(f"- {ticker}: Stop-Loss {stop_loss}, Take-Profit {take_profit}"))

    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin-1')
    elif isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)
    return pdf_bytes
