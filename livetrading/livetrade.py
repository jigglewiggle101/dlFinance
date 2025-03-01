import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API credentials from .env file
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets'  # Corrected endpoint for paper trading

# Initialize Alpaca API client
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Function to place an order
def place_order(symbol, qty, side='buy', order_type='market', time_in_force='gtc'):
    try:
        # Place the order
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

# Example of placing a buy order
place_order('AAPL', 10)  # Buys 10 shares of AAPL
