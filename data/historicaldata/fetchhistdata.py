import yfinance as yf
import os
import time
import pandas as pd

# Function to fetch historical data for multiple symbols
def fetch_historical_data(symbols, start='2020-01-01', end='2021-01-01', interval='1d', save_to_csv=True):
    # Check if the directory exists, if not, create it
    directory = 'data/historical_data'
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist
    
    for symbol in symbols:
        try:
            print(f"Fetching data for {symbol}...")
            data = yf.download(symbol, start=start, end=end, interval=interval)
            
            # Print the data to the console (optional)
            print(data.head())  # Show the first few rows for verification
            
            # Save data to CSV file
            if save_to_csv:
                file_name = f"{symbol}_historical_data_{start}_to_{end}.csv"
                file_path = os.path.join(directory, file_name)  # Save it in the 'historical_data' folder
                data.to_csv(file_path)
                print(f"Data for {symbol} saved to {file_path}")
            
            time.sleep(1)  # Adding sleep to avoid hitting rate limits on Yahoo Finance
        except Exception as e:
            print(f"Could not fetch data for {symbol}: {str(e)}")
    
    return data

# List of publicly traded companies (replace or add more tickers as needed)
symbols = ['AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA']

# Fetch historical data for these symbols between 2020 and 2021
fetch_historical_data(symbols, start='2020-01-01', end='2021-01-01', interval='1d')
