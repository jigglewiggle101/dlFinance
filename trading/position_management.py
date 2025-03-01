# Position sizing, stop-loss, take-profit, etc.

def calculate_position_size(account_balance, risk_percentage, stop_loss_distance):
    risk_per_trade = account_balance * risk_percentage
    position_size = risk_per_trade / stop_loss_distance
    return position_size
