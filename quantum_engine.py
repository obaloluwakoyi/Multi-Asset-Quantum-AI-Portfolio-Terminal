# quantum_engine.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.circuit.library import EfficientSU2
    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

def fetch_market_data(tickers_tuple):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    raw = yf.download(list(tickers_tuple), start=start_date, end=end_date, auto_adjust=True, progress=False)
    
    data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    data = data.dropna(axis=1, how="all")
    returns = data.pct_change().dropna()
    
    mu = returns.mean().values
    sigma = returns.cov().values
    return mu, sigma, returns, data

def run_quantum_optimization(mu, sigma, valid_tickers, risk_val, q_budget):
    if not QISKIT_AVAILABLE:
        return _classical_fallback(mu, q_budget, valid_tickers)
        
    try:
        q_risk = max(1, min(10, risk_val)) / 10.0
        budget = min(q_budget, len(valid_tickers))
        portfolio_problem = PortfolioOptimization(
            expected_returns=mu, covariances=sigma, risk_factor=q_risk, budget=budget
        )
        qp = portfolio_problem.to_quadratic_program()
        ansatz = EfficientSU2(qp.get_num_vars(), reps=2, entanglement="full")
        vqe = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=COBYLA(maxiter=200))
        result = MinimumEigenOptimizer(vqe).solve(qp)
        return portfolio_problem.interpret(result)
    except Exception:
        return _classical_fallback(mu, q_budget, valid_tickers)

def _classical_fallback(mu, q_budget, valid_tickers):
    n_select = min(q_budget, len(valid_tickers))
    top_idx = np.argsort(mu)[-n_select:][::-1]
    selection = np.zeros(len(valid_tickers), dtype=int)
    for idx in top_idx:
        selection[idx] = max(1, q_budget // n_select)
    return selection

def run_monte_carlo(port_daily, investment_total, mc_paths, horizon=252):
    daily_mu = port_daily.mean()
    daily_std = port_daily.std()
    simulations = np.zeros((horizon, mc_paths))
    for i in range(mc_paths):
        shocks = np.random.normal(daily_mu, daily_std, horizon)
        simulations[:, i] = investment_total * (1 + shocks).cumprod()
    return simulations