# Main trading strategy logic (buy/sell/hold decisions)

import alpaca_trade_api as tradeapi
import os

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def place_order(symbol, qty, side='buy', order_type='market', time_in_force='gtc'):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        print(f"Order placed: {side} {qty} {symbol}")
    except Exception as e:
        print(f"Error placing order: {e}")

# Example usage: buy 10 AAPL shares
place_order('AAPL', 10)
