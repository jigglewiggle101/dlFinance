import alpaca_trade_api as tradeapi
import os

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def fetch_real_time_data(symbol='AAPL', limit=100):
    barset = api.get_barset(symbol, 'minute', limit=limit)
    bars = barset[symbol]
    for bar in bars:
        print(f'Time: {bar.t} | Open: {bar.o} | High: {bar.h} | Low: {bar.l} | Close: {bar.c} | Volume: {bar.v}')
    return bars

# Example of usage
fetch_real_time_data('AAPL', limit=10)
