import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API credentials from .env file
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets'  # Corrected endpoint

# Initialize Alpaca API client
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Function to fetch real-time data for a given symbol (AAPL, for example)
def fetch_real_time_data(symbol='AAPL', limit=100):
    try:
        # Fetch real-time bar data (1-minute intervals)
        bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit).df

        # Check if any data was returned
        if bars.empty:
            print("No data returned")
        else:
            # Print the data to check the result
            for _, bar in bars.iterrows():
                print(f"Time: {bar.name} | Open: {bar.open} | High: {bar.high} | Low: {bar.low} | Close: {bar.close} | Volume: {bar.volume}")
        
        return bars
    except Exception as e:
        print(f"Error occurred: {e}")

# Example of usage: Fetch real-time data for AAPL, limit to the last 10 bars
fetch_real_time_data('AAPL', limit=10)
