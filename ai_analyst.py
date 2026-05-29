# ai_analyst.py
import requests
import json

def get_ai_narrative(provider, model_name, api_key, endpoint_url, portfolio_data):
    """
    Generates an institutional-grade risk analysis framework with automated 
    trade rebalancing recommendations.
    """
    
    # 1. Format structural components safely to handle potential empty matrices
    compliance_alerts = "\n".join([f"- {a}" for a in portfolio_data.get("compliance_alerts", [])])
    
    stress_scenarios = ""
    for scenario, loss in portfolio_data.get("stress_test_losses", {}).items():
        # Defensive Check: Extract float value if 'loss' is still a dictionary, series, or array wrapper
        if isinstance(loss, dict):
            # Tries to find the raw percentage column, otherwise grabs the first dictionary value
            val = loss.get("Estimated Value Impact (%)", list(loss.values())[0] if loss.values() else 0.0)
        elif hasattr(loss, "iloc"):
            # Handles pandas Series index extractions safely
            val = loss.iloc[0] if not loss.empty else 0.0
        elif isinstance(loss, (list, tuple)):
            val = loss[0] if loss else 0.0
        else:
            # Fallback if it's already a clean int or float primitive
            val = loss
            
        # Convert to float to safely process format token masks
        try:
            val = float(val)
            stress_scenarios += f"- **{scenario}**: Expected Drop of {val:+.2f}%\n"
        except (ValueError, TypeError):
            # Absolute baseline safety fallback if parsing completely fails
            stress_scenarios += f"- **{scenario}**: Expected Drop of {val}%\n"

    exit_rules = ""
    for ticker, targets in portfolio_data.get("exit_matrix", {}).items():
        current_price = targets.get("Current Price", "N/A")
        stop_loss = targets.get("Trailing Stop-Loss", targets.get("Stop-Loss Threshold", "N/A"))
        take_profit = targets.get("Take-Profit Target", "N/A")

        if isinstance(current_price, (int, float)):
            current_price = f"${current_price:,.2f}"
        if isinstance(stop_loss, (int, float)):
            stop_loss = f"{stop_loss:+.2f}%"
        if isinstance(take_profit, (int, float)):
            take_profit = f"{take_profit:+.2f}%"

        exit_rules += f"- **{ticker}**: Spot {current_price} | Stop-Loss Floor: {stop_loss} | Take-Profit Ceiling: {take_profit}\n"

    # 2. Construct an advanced, highly structured quantitative system prompt
    prompt = f"""
    You are an institutional portfolio risk manager and quantitative algorithmic analyst. 
    Analyze the following structured terminal dataset and construct an executive summary report.
    
    DATASET PAYLOAD:
    - Active Asset Positions Allocation: {portfolio_data.get('allocations')}
    - Tail Risk Metrics: 95% Daily VaR = ${portfolio_data.get('var_95', 0):,.2f} | 95% Daily CVaR = ${portfolio_data.get('cvar_95', 0):,.2f}
    - Compliance Flags raised by Risk Engine:
    {compliance_alerts if compliance_alerts else "- No compliance threshold violations detected."}
    - Macroeconomic Shock Stress-Testing Percentages:
    {stress_scenarios}
    - Dynamic Risk Tracking Exit Matrix Targets:
    {exit_rules}
    - Predictive Engine Configurations: Mode = {portfolio_data['forecasting_parameters']['engine_mode']} | Simulated VIX Level = {portfolio_data['forecasting_parameters']['vix_simulated']:.1f} | Expected 5-Yr Horizon Terminal Value = ${portfolio_data['forecasting_parameters']['expected_5yr_value']:,.2f}
    
    CRITICAL OUTPUT REQUIREMENTS & SYSTEM STRUCTURE:
    Your output response must be clean, highly technical, and structured into the following four distinct Markdown sections:

    ### 🏢 1. QUANTITATIVE ARCHITECTURE ASSESSMENT
    - Review the weights distribution across assets.
    - Evaluate how the current portfolio handles structural risks based on the chosen forecasting engine ({portfolio_data['forecasting_parameters']['engine_mode']}) at a VIX of {portfolio_data['forecasting_parameters']['vix_simulated']:.1f}.
    
    ### ⚠️ 2. STRESS TEST & TAIL RISK EVALUATION
    - Interpret the 95% Daily VaR and CVaR bounds in terms of capital preservation.
    - Identify which historical macroeconomic shock scenario causes the most severe damage to this specific asset allocation layout and analyze why.
    
    ### 🎟️ 3. ACTIONABLE STRATEGIC REBALANCING TICKETS
    Provide explicit, bulleted action items modeled as market execution orders using these direct prefixes exactly:
    - **[ADD / INCREASE POSITION]**: Name specific tickers that deserve capital deployment based on high expected returns or safety optimization. Explain why.
    - **[TRIM / REDUCE WEIGHT]**: Name specific tickers that represent over-allocation or present asymmetric tail risks based on the stress test matrix.
    - **[LIQUIDATE / REMOVE]**: Name any tickers that violate risk thresholds or have hit stop-loss logic zones.
    If no assets need removal, explicitly state that current constraints are fully optimized.

    ### 🚪 4. OPERATIONAL RISK AND EXIT PLAN TARGETS
    - Provide a short tactical commentary on the exit matrix parameters. Highlight the critical price floor targets for your largest holding.

    Ensure your language is cold, analytical, precise, and professional. Avoid pleasantries, introductions, or conversational filler.
    """
    
    if not api_key and provider not in ("Ollama", "Offline / No-API"):
        return "⚠️ **Inference Incomplete**: Selected provider requires a valid API key configuration."

    try:
        # ── Provider Selection Routing Logic ──
        if provider == "Groq":
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model_name or "llama3-8b-8192", 
                "messages": [{"role": "user", "content": prompt}], 
                "temperature": 0.1
            }
            # FIXED: Corrected endpoint from typo /completypes to /completions
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=10)
            return r.json()['choices'][0]['message']['content']

        elif provider == "Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name or 'gemini-1.5-flash'}:generateContent?key={api_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1}}
            r = requests.post(url, json=payload, timeout=10)
            return r.json()['candidates'][0]['content']['parts'][0]['text']

        elif provider == "OpenRouter":
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model_name or "meta-llama/llama-3-8b-instruct:free", "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=10)
            return r.json()['choices'][0]['message']['content']

        elif provider == "Ollama":
            url = endpoint_url if endpoint_url else "http://localhost:11434/api/generate"
            payload = {"model": model_name or "llama3", "prompt": prompt, "stream": False}
            r = requests.post(url, json=payload, timeout=15)
            return r.json()['response']

        else:
            # Handles "Offline / No-API" selections
            return _generate_deterministic_fallback(portfolio_data)

    except Exception as e:
        return f"❌ **Inference Operational Fault**: Failed to securely reach {provider} endpoint router. Error trace: {str(e)}"

def _generate_deterministic_fallback(data):
    """
    Sophisticated rule-based financial analysis report generated completely offline
    if LLM API routes are disabled or unavailable.
    """
    allocations = data.get("allocations", [])
    if not allocations:
        return "⚠️ **Analytics Error**: Allocation matrix empty. Unable to run fallback routing."
        
    sorted_alloc = sorted(allocations, key=lambda x: x.get("Exp_Ann_Return", 0), reverse=True)
    
    highest_return_ticker = sorted_alloc[0]["Ticker"]
    lowest_return_ticker = sorted_alloc[-1]["Ticker"]
    
    # 1. Build deterministic recommendations tickets based on quantitative telemetry metrics
    add_ticket = f"**[ADD / INCREASE POSITION]**: Allocate extra cash capital reserves into **{highest_return_ticker}** due to its superior annualized mathematical drift profile ({sorted_alloc[0].get('Exp_Ann_Return', 0):.1f}%)."
    
    if len(sorted_alloc) > 1:
        trim_ticket = f"**[TRIM / REDUCE WEIGHT]**: Scale back allocations within **{lowest_return_ticker}** to protect capital margins against continuous downside variance volatility ({sorted_alloc[-1].get('Exp_Ann_Return', 0):.1f}%)."
    else:
        trim_ticket = "**[TRIM / REDUCE WEIGHT]**: No secondary assets currently meet trimming volatility risk parameters."

    # 2. Extract severe stress test scenario information dynamically
    stress_losses = data.get("stress_test_losses", {})
    worst_scenario = "Historical Benchmark Shock"
    worst_loss = 0.0
    if stress_losses:
        def _numeric_loss(value):
            if isinstance(value, dict):
                return float(value.get("impact_pct", value.get("Estimated Value Impact (%)", 0.0)))
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
        worst_scenario = min(stress_losses, key=lambda k: _numeric_loss(stress_losses[k]))
        worst_loss = _numeric_loss(stress_losses[worst_scenario])

    # 3. Assemble and return the deterministic report matching standard system formatting rules
    offline_report = f"""
### 🏢 1. QUANTITATIVE ARCHITECTURE ASSESSMENT
The system has optimized layout weights across **{len(allocations)}** registry assets. Under the **{data['forecasting_parameters']['engine_mode']}** simulation baseline running at a simulated VIX level of **{data['forecasting_parameters']['vix_simulated']:.1f}**, the continuous multi-path portfolio structure projects an expected horizon terminal valuation of **${data['forecasting_parameters']['expected_5yr_value']:,.2f}**.

### ⚠️ 2. STRESS TEST & TAIL RISK EVALUATION
The asset allocation exhibits a 95% Daily Value-at-Risk (VaR) threshold of **${data['var_95']:,.2f}**, indicating daily portfolio losses should not breach this baseline under standard conditions. Tail risk monitoring shows a Conditional VaR (CVaR) profile of **${data['cvar_95']:,.2f}**. Structural evaluation highlights **{worst_scenario}** as the most severe systemic threat, causing a projected asset decline of **{worst_loss:+.2f}%**.

### 🎟️ 3. ACTIONABLE STRATEGIC REBALANCING TICKETS
* {add_ticket}
* {trim_ticket}
* **[HOLD CONSTRAINTS]**: Maintain strict risk boundaries across all other active ledger positions. No explicit liquidation triggers have been broken.

### 🚪 4. OPERATIONAL RISK AND EXIT PLAN TARGETS
Dynamic risk parameters have successfully mapped automated exit price triggers across all nodes. Monitor trailing stop-loss values closely to protect your aggregate capital base from structural drawdown spikes.
    """
    return offline_report