import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alpaca API credentials from environment variables
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets/v2'  # For paper trading

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def place_order(symbol, qty, side='buy', order_type='market', time_in_force='gtc'):
    try:
        # Fetch account details to check funds
        account = api.get_account()  
        print(f"Available Cash: {account.cash}")

        # Check if there's enough cash for the order
        if float(account.cash) < qty * 125:  # Assuming 125 is the price per share
            print(f"Not enough cash to place order for {qty} shares of {symbol}.")
            return

        # Submit the order
        response = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )

        # Check the response for any issues
        print(f"Order response: {response}")
        
    except Exception as e:
        print(f"Error placing order: {e}")

# Example of placing a buy order
place_order('NVDA', 1)
