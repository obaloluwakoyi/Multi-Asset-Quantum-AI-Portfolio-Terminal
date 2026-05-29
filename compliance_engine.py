# compliance_engine.py
import numpy as np
import pandas as pd

def run_compliance_audit(df_report, returns, investment_total):
    alerts = []
    weights_dict = df_report.set_index("Ticker")["Weight"].to_dict()
    
    # Concentrated Asset Risk Flag
    for ticker, weight in weights_dict.items():
        if weight > 0.40:
            alerts.append(f"CRITICAL OVER-EXPOSURE: Asset {ticker} represents {weight:.1%} of portfolio capital.")
            
    # Tail-Risk Assessment (95% Daily VaR / CVaR)
    port_daily = returns[df_report["Ticker"]].mul(df_report.set_index("Ticker")["Weight"], axis=1).sum(axis=1)
    var_95 = np.percentile(port_daily, 5) * investment_total
    cvar_95 = port_daily[port_daily <= np.percentile(port_daily, 5)].mean() * investment_total
    
    return {
        "alerts": alerts if alerts else ["All nominal weight compliance markers checked and validated."],
        "var_95": abs(var_95),
        "cvar_95": abs(cvar_95)
    }

def execute_stress_scenarios(df_report, returns, investment_total):
    """Evaluates historic or structural market impacts on the current allocation weights."""
    port_daily = returns[df_report["Ticker"]].mul(df_report.set_index("Ticker")["Weight"], axis=1).sum(axis=1)
    
    scenarios = {
        "2008 Lehman Systemic Meltdown": -0.045,
        "2020 Liquidity Deliquation (COVID)": -0.052,
        "Tech Valuation Compress Event": -0.028,
        "Rate Hike Inflationary Shock": -0.015
    }
    
    results = {}
    for name, shock in scenarios.items():
        daily_vol = port_daily.std()
        estimated_impact = (shock * 1.5) + (daily_vol * -2.0)
        results[name] = {
            "impact_pct": estimated_impact * 100,
            "capital_risk": investment_total * (1 + estimated_impact)
        }
    return results

def generate_dynamic_exit_plan(df_report, returns):
    """Calculates algorithmic take-profit and stop-loss criteria using localized volatility profiles."""
    exit_strategies = {}
    for _, row in df_report.iterrows():
        ticker = row["Ticker"]
        ticker_vol = returns[ticker].std() * np.sqrt(252) # Annualized Volatility
        
        # Scale exit parameters adaptively relative to the asset's volatility profile
        stop_loss_pct = max(0.08, min(0.25, ticker_vol * 0.5))
        take_profit_pct = max(0.15, min(0.60, ticker_vol * 1.5))
        
        exit_strategies[ticker] = {
            "Stop-Loss Threshold": f"-{stop_loss_pct:.1%}",
            "Take-Profit Target": f"+{take_profit_pct:.1%}",
            "Rebalancing Horizon": "Quarterly Adjust" if ticker_vol < 0.3 else "Bi-Weekly Monitoring Required"
        }
    return exit_strategies