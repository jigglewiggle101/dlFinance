# Risk management functions (e.g., max drawdown, diversification)

def risk_management(account_balance, max_drawdown_percentage):
    drawdown_limit = account_balance * max_drawdown_percentage
    return drawdown_limit
