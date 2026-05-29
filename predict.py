# predict.py
import numpy as np
import pandas as pd

def run_gbm_paths(port_daily, initial_capital, paths, horizon_days=252):
    """Generates forecasting pathways using standard Geometric Brownian Motion."""
    mu = port_daily.mean()
    sigma = port_daily.std()
    
    # Drift and Diffusion coefficients
    dt = 1 / 252
    drift = (mu - 0.5 * (sigma ** 2)) * dt
    diffusion = sigma * np.sqrt(dt)
    
    simulations = np.zeros((horizon_days, paths))
    simulations[0, :] = initial_capital
    
    for t in range(1, horizon_days):
        shocks = np.random.normal(0, 1, paths)
        simulations[t, :] = simulations[t-1, :] * np.exp(drift + diffusion * shocks)
        
    return simulations

def run_merton_jump_paths(port_daily, initial_capital, paths, horizon_days=252):
    """Generates forecasting pathways adding Poisson jump-diffusion profiles."""
    mu = port_daily.mean()
    sigma = port_daily.std()
    
    # Structural Jump Parameters (Calibrated to represent sudden market shock vulnerabilities)
    lambda_j = 0.1     # Intensity parameter (Probability of a structural jump occurring per year)
    mu_j = -0.15       # Expected magnitude of the jump (e.g., a sudden 15% drop)
    sigma_j = 0.08     # Standard deviation/variance of the jump size
    
    dt = 1 / 252
    k = np.exp(mu_j + 0.5 * (sigma_j ** 2)) - 1
    
    # Adjusted continuous drift component matching structural martingale constraints
    drift = (mu - 0.5 * (sigma ** 2) - lambda_j * k) * dt
    diffusion = sigma * np.sqrt(dt)
    
    simulations = np.zeros((horizon_days, paths))
    simulations[0, :] = initial_capital
    
    for t in range(1, horizon_days):
        # Continuous Gaussian shock components
        z_diffusion = np.random.normal(0, 1, paths)
        
        # Poisson counter for sudden regime jumps
        n_jumps = np.random.poisson(lambda_j * dt, paths)
        jump_impact = np.zeros(paths)
        
        for p in range(paths):
            if n_jumps[p] > 0:
                # Accumulate multi-frequency normal jump distributions if multiple events strike
                jump_impact[p] = np.sum(np.random.normal(mu_j, sigma_j, n_jumps[p]))
        
        simulations[t, :] = simulations[t-1, :] * np.exp(drift + diffusion * z_diffusion + jump_impact)
        
    return simulations

def run_adaptive_forecasting_engine(port_daily, initial_capital, paths, mode, blend_ratio=0.8, vix_value=18.0):
    """
    Executes advanced predictive simulation matrices.
    Modes: 'Pure GBM', 'Pure Merton', 'Dynamic Blend', 'Trigger-Switch (VIX)'
    """
    horizon_days = 252 * 5 # Scale forward to compile a 5-Year target horizon framework
    
    if mode == "Pure GBM":
        return run_gbm_paths(port_daily, initial_capital, paths, horizon_days)
        
    elif mode == "Pure Merton":
        return run_merton_jump_paths(port_daily, initial_capital, paths, horizon_days)
        
    elif mode == "Dynamic Blend":
        gbm_paths = run_gbm_paths(port_daily, initial_capital, paths, horizon_days)
        merton_paths = run_merton_jump_paths(port_daily, initial_capital, paths, horizon_days)
        # Structural asset vector blending matrix
        return (blend_ratio * gbm_paths) + ((1 - blend_ratio) * merton_paths)
        
    elif mode == "Trigger-Switch (VIX)":
        # Critical structural risk indicator evaluation threshold
        if vix_value >= 25.0:
            return run_merton_jump_paths(port_daily, initial_capital, paths, horizon_days)
        else:
            return run_gbm_paths(port_daily, initial_capital, paths, horizon_days)

def compile_year_target_table(simulations, initial_capital):
    """Compiles multi-year target projections evaluating timeline thresholds."""
    steps_per_year = 252
    years = [1, 2, 3, 4, 5]
    records = []
    
    for y in years:
        step_idx = (y * steps_per_year) - 1
        # Prevent tracking index out-of-bounds metrics
        if step_idx >= len(simulations):
            step_idx = len(simulations) - 1
            
        year_end_values = simulations[step_idx, :]
        mean_val = year_end_values.mean()
        p10 = np.percentile(year_end_values, 10)
        p90 = np.percentile(year_end_values, 90)
        
        # Calculate localized total compound performance metrics
        cagr = ((mean_val / initial_capital) ** (1 / y) - 1) * 100
        
        records.append({
            "Target Timeline": f"Year {y} Horizon",
            "Expected Value ($)": mean_val,
            "Downside Bound (P10) ($)": p10,
            "Upside Bound (P90) ($)": p90,
            "Implied Return Engine (CAGR)": f"{cagr:+.2f}%"
        })
        
    return pd.DataFrame(records)