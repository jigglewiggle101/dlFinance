import tweepy
import pandas as pd
from textblob import TextBlob
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Twitter API credentials from environment variables
API_KEY = os.getenv('TWITTER_API_KEY')
API_SECRET_KEY = os.getenv('TWITTER_API_SECRET_KEY')
ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

# Set up the tweepy client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# List of stock symbols to fetch sentiment for
symbols = ['AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA']

# Function to fetch tweets for a given stock symbol
def fetch_tweets(symbol, max_results=10):
    tweets = []
    try:
        # Search for tweets related to the symbol
        response = client.search_recent_tweets(query=symbol, max_results=max_results)
        for tweet in response.data:
            tweets.append(tweet.text)
        print(f"Fetched {len(tweets)} tweets for {symbol}")
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
    return tweets

# Function to analyze sentiment of tweets using TextBlob
def analyze_sentiment(tweets):
    sentiment_scores = []
    for tweet in tweets:
        # Perform sentiment analysis using TextBlob
        analysis = TextBlob(tweet)
        sentiment_scores.append(analysis.sentiment.polarity)
    return sentiment_scores

# Function to fetch and analyze sentiment for a list of stocks in batches
def fetch_sentiment_data(symbols, batch_size=10):
    sentiment_data = []

    # Process symbols in batches
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}...")

        for symbol in batch_symbols:
            print(f"Fetching tweets for {symbol}...")
            tweets = fetch_tweets(symbol)
            if not tweets:
                print(f"No tweets found for {symbol}")
                continue

            # Analyze the sentiment of the fetched tweets
            sentiment_scores = analyze_sentiment(tweets)

            # Collect sentiment data
            sentiment_data.append({
                'symbol': symbol,
                'tweets': len(tweets),
                'positive_sentiment': sum(1 for score in sentiment_scores if score > 0),
                'negative_sentiment': sum(1 for score in sentiment_scores if score < 0),
                'neutral_sentiment': sum(1 for score in sentiment_scores if score == 0),
                'avg_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            })

    # Convert sentiment data to a DataFrame
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df.to_csv('sentiment_analysis.csv', index=False)
    print(f"Sentiment data saved to 'sentiment_analysis.csv'")

# Fetch and analyze sentiment data for the specified symbols
fetch_sentiment_data(symbols)

