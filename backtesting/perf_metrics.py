# Performance evaluation metrics (Sharpe ratio, Win rate, etc.)

import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)
