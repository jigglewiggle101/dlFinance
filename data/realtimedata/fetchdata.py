import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API credentials
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets/v2'  # Corrected endpoint

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')



# def fetch_real_time_data(symbol='AAPL', limit=100):
#     try:
#         # Fetching real-time bar data (1-minute intervals)
#         bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit).df

#         # Print the data to check the result
#         if bars.empty:
#             print("No data returned")
#         else:
#             for _, bar in bars.iterrows():
#                 print(f'Time: {bar.name} | Open: {bar.open} | High: {bar.high} | Low: {bar.low} | Close: {bar.close} | Volume: {bar.volume}')
        
#         return bars
#     except Exception as e:
#         print(f"Error occurred: {e}")

# # Example of usage
# fetch_real_time_data('AAPL', limit=10)
