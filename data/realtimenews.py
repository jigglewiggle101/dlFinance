import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
import time
import logging
from typing import Dict, List, Optional, Union, Tuple
import concurrent.futures
from functools import lru_cache
from pathlib import Path
import ta  # Technical analysis library for indicators
from scipy import stats
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("corporate_actions_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CorporateActions")


class CorporateActionCollector:
    """Optimized collector for corporate actions with advanced technical analysis"""
    
    def __init__(self, api_key: str = None, cache_dir: str = "cache"):
        """
        Initialize with API credentials and caching configuration
        
        Args:
            api_key: API key for financial data provider (default uses env variable)
            cache_dir: Directory to cache API responses
        """
        self.api_key = api_key or os.environ.get("FINANCIAL_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set FINANCIAL_API_KEY environment variable")
        
        # Alpha Vantage is used as example, but code supports multiple providers
        self.primary_api = {
            "name": "alphavantage",
            "base_url": "https://www.alphavantage.co/query",
            "rate_limit": 5,  # calls per minute
            "rate_window": 60  # seconds
        }
        
        # Backup API if available
        self.backup_api = os.environ.get("BACKUP_API_KEY") and {
            "name": "polygon",
            "base_url": "https://api.polygon.io/v3",
            "rate_limit": 5,
            "rate_window": 60
        }
        
        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_expiry = 86400  # 24 hours in seconds
        
        # Track API usage
        self.last_call_time = datetime.now()
        self.call_count = 0
    
    @lru_cache(maxsize=100)
    def _get_cached_response(self, endpoint: str, params_key: str) -> Optional[Dict]:
        """Get cached API response if available and not expired"""
        cache_file = self.cache_dir / f"{endpoint}_{params_key}.json"
        
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < self.cache_expiry:
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid cache file: {cache_file}")
        
        return None
    
    def _cache_response(self, endpoint: str, params_key: str, data: Dict) -> None:
        """Cache API response for future use"""
        cache_file = self.cache_dir / f"{endpoint}_{params_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def _api_call(self, endpoint: str, params: Dict) -> Dict:
        """Make API call with rate limiting and caching"""
        # Generate cache key from params
        params_key = '_'.join(f"{k}_{v}" for k, v in sorted(params.items()) if k != "apikey")
        
        # Check cache first
        cached_data = self._get_cached_response(endpoint, params_key)
        if cached_data:
            return cached_data
        
        # Rate limiting
        current_time = datetime.now()
        elapsed = (current_time - self.last_call_time).total_seconds()
        
        if elapsed < self.primary_api["rate_window"] and self.call_count >= self.primary_api["rate_limit"]:
            sleep_time = self.primary_api["rate_window"] - elapsed
            logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            self.call_count = 0
            self.last_call_time = datetime.now()
        
        # Make the actual API call
        try:
            params["apikey"] = self.api_key
            response = requests.get(f"{self.primary_api['base_url']}/{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Update call tracking
            self.call_count += 1
            if elapsed >= self.primary_api["rate_window"]:
                self.call_count = 1
                self.last_call_time = current_time
            
            # Cache the response
            self._cache_response(endpoint, params_key, data)
            
            return data
            
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            
            # Try backup API if available
            if self.backup_api:
                try:
                    logger.info("Trying backup API...")
                    # Implement backup API call logic here
                    # ...
                    
                except Exception as e2:
                    logger.error(f"Backup API also failed: {e2}")
            
            return {}
    
    def get_price_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical price data for technical analysis
        
        Returns DataFrame with OHLCV data
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full"
        }
        
        data = self._api_call("", params)
        
        if "Time Series (Daily)" not in data:
            logger.warning(f"No price data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        price_data = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        
        # Rename columns
        price_data.columns = [col.split('. ')[1] for col in price_data.columns]
        
        # Convert to numeric
        for col in price_data.columns:
            price_data[col] = pd.to_numeric(price_data[col])
        
        # Filter by date range
        price_data = price_data.loc[start_date:end_date]
        
        # Ensure data is sorted by date
        price_data = price_data.sort_index()
        
        # Rename for consistent column names
        price_data.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'adjusted close': 'adj_close'
        }, inplace=True)
        
        return price_data
    
    def get_dividends(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get dividend history for a stock
        
        Returns a DataFrame with dividend information
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full"
        }
        
        data = self._api_call("", params)
        
        dividends = []
        if "Time Series (Daily)" in data:
            for date, daily_data in data["Time Series (Daily)"].items():
                if start_date <= date <= end_date:
                    dividend_amount = float(daily_data.get("7. dividend amount", 0))
                    if dividend_amount > 0:
                        dividends.append({
                            "symbol": symbol,
                            "date": date,
                            "amount": dividend_amount,
                            "currency": "USD",
                            "action_type": "dividend"
                        })
        
        return pd.DataFrame(dividends) if dividends else pd.DataFrame(
            columns=["symbol", "date", "amount", "currency", "action_type"]
        )
    
    def get_splits(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get stock split history
        
        Returns a DataFrame with split information
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full"
        }
        
        data = self._api_call("", params)
        
        splits = []
        if "Time Series (Daily)" in data:
            # Convert to DataFrame for easier analysis
            ts_data = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            ts_data = ts_data.sort_index()
            
            # Filter by date range
            ts_data = ts_data.loc[start_date:end_date]
            
            # Convert columns to numeric
            for col in ts_data.columns:
                ts_data[col] = pd.to_numeric(ts_data[col])
            
            # Calculate ratios between raw close and adjusted close
            ts_data['ratio'] = ts_data['4. close'] / ts_data['5. adjusted close']
            
            # Calculate day-to-day changes in the ratio
            ts_data['ratio_change'] = ts_data['ratio'].pct_change()
            
            # Identify significant changes (potential splits)
            # Using a threshold that will catch most splits (a 5% change in the ratio)
            split_days = ts_data[abs(ts_data['ratio_change']) > 0.05].index
            
            for day in split_days:
                # Calculate actual split ratio
                if day > ts_data.index[0]:  # Skip first day (no previous day)
                    prev_day = ts_data.index[ts_data.index.get_loc(day) - 1]
                    prev_ratio = ts_data.loc[prev_day, 'ratio']
                    curr_ratio = ts_data.loc[day, 'ratio']
                    
                    # Determine split ratio (e.g., 2:1, 3:1, 1:10)
                    ratio = curr_ratio / prev_ratio
                    
                    if ratio > 1:
                        split_ratio = f"{round(ratio, 1)}:1"  # e.g., 2:1 for a 2-for-1 split
                    else:
                        split_ratio = f"1:{round(1/ratio, 1)}"  # e.g., 1:2 for a reverse split
                    
                    splits.append({
                        "symbol": symbol,
                        "date": day,
                        "ratio": split_ratio,
                        "factor": ratio,
                        "action_type": "split"
                    })
        
        return pd.DataFrame(splits) if splits else pd.DataFrame(
            columns=["symbol", "date", "ratio", "factor", "action_type"]
        )
    
    def get_mergers(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get merger and acquisition data
        
        May require specialized financial news API.
        Returns a DataFrame with merger information.
        """
        # This would typically use a financial news API
        # For demonstration, we'll return a placeholder DataFrame
        
        # Convert dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # In a real implementation, you would fetch this data from a specialized source
        # For now, just check if we have any hardcoded known mergers for this symbol
        known_mergers = {
            "LNKD": [{"date": "2016-06-13", "acquirer": "MSFT", "deal_value": 26200000000}],
            "TWTR": [{"date": "2022-10-28", "acquirer": "Private", "deal_value": 44000000000}],
            # Add more known mergers here
        }
        
        mergers = []
        if symbol in known_mergers:
            for merger in known_mergers[symbol]:
                merger_date = datetime.strptime(merger["date"], '%Y-%m-%d')
                if start_dt <= merger_date <= end_dt:
                    mergers.append({
                        "symbol": symbol,
                        "date": merger["date"],
                        "acquirer": merger["acquirer"],
                        "deal_value": merger["deal_value"],
                        "action_type": "merger"
                    })
        
        return pd.DataFrame(mergers) if mergers else pd.DataFrame(
            columns=["symbol", "date", "acquirer", "deal_value", "action_type"]
        )
    
    def analyze_technical_indicators(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for the price data
        
        Args:
            price_df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if price_df.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = price_df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns for technical analysis: {missing_cols}")
            return df
        
        try:
            # VOLUME-BASED INDICATORS
            # On-Balance Volume (OBV)
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
            
            # Chaikin Money Flow (CMF)
            df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20
            ).chaikin_money_flow()
            
            # Accumulation/Distribution Line
            df['adl'] = ta.volume.AccumulationDistributionIndicator(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
            ).acc_dist_index()
            
            # Volume Weighted Average Price (VWAP)
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            # MOMENTUM AND TREND STRENGTH INDICATORS
            # Average Directional Index (ADX)
            adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = adx_indicator.adx()
            df['adx_pos'] = adx_indicator.adx_pos()
            df['adx_neg'] = adx_indicator.adx_neg()
            
            # Commodity Channel Index (CCI)
            df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r()
            
            # VOLATILITY INDICATORS
            # Average True Range (ATR)
            df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bollinger_mavg'] = bollinger.bollinger_mavg()
            df['bollinger_hband'] = bollinger.bollinger_hband()
            df['bollinger_lband'] = bollinger.bollinger_lband()
            df['bollinger_width'] = (df['bollinger_hband'] - df['bollinger_lband']) / df['bollinger_mavg']
            
            # OTHER TECHNICAL TOOLS
            # Moving Averages
            df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['sma_200'] = ta.trend.SMAIndicator(close=df['close'], window=200).sma_indicator()
            df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
            
            # MACD
            macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            
            # Ichimoku Cloud (simplified components)
            ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
            
            # Pivot Points (using previous day's data)
            df['pivot_point'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
            df['pivot_r1'] = 2 * df['pivot_point'] - df['low'].shift(1)
            df['pivot_s1'] = 2 * df['pivot_point'] - df['high'].shift(1)
            
            # Rate of Change
            df['roc'] = ta.momentum.ROCIndicator(close=df['close'], window=12).roc()
            
            # Fibonacci retracement levels are typically calculated for specific price moves and displayed as horizontal lines
            
            logger.info(f"Calculated technical indicators (total: {len(df.columns) - len(price_df.columns)})")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return price_df
    
    def analyze_corporate_action_impact(
        self, 
        symbol: str, 
        price_df: pd.DataFrame, 
        actions_df: pd.DataFrame
    ) -> Dict:
        """
        Analyze the impact of corporate actions on price and technical indicators
        
        Args:
            symbol: Stock symbol
            price_df: DataFrame with price history and technical indicators
            actions_df: DataFrame with corporate actions
            
        Returns:
            Dictionary with impact analysis results
        """
        if price_df.empty or actions_df.empty:
            return {"impact_analysis": "No data available for analysis"}
        
        # Filter actions for this symbol
        symbol_actions = actions_df[actions_df['symbol'] == symbol]
        
        if symbol_actions.empty:
            return {"impact_analysis": f"No corporate actions found for {symbol}"}
        
        # Convert dates to datetime for easier comparison
        if not pd.api.types.is_datetime64_dtype(price_df.index):
            price_df.index = pd.to_datetime(price_df.index)
            
        if not pd.api.types.is_datetime64_dtype(symbol_actions['date']):
            symbol_actions['date'] = pd.to_datetime(symbol_actions['date'])
        
        # Results container
        impact_results = {
            "symbol": symbol,
            "total_actions": len(symbol_actions),
            "action_types": symbol_actions['action_type'].value_counts().to_dict(),
            "actions": []
        }
        
        # Analyze each action
        for _, action in symbol_actions.iterrows():
            action_date = action['date']
            action_type = action['action_type']
            
            # Get price data around the event (10 days before, 10 days after if available)
            pre_event_start = action_date - pd.Timedelta(days=20)
            post_event_end = action_date + pd.Timedelta(days=20)
            
            event_window = price_df.loc[pre_event_start:post_event_end] if pre_event_start in price_df.index else pd.DataFrame()
            
            if event_window.empty or len(event_window) < 5:
                # Not enough data around the event
                continue
            
            # Split into pre and post event
            pre_event = event_window.loc[:action_date].iloc[:-1]  # Exclude event day
            post_event = event_window.loc[action_date:].iloc[1:]  # Exclude event day
            
            # Skip if not enough data
            if len(pre_event) < 3 or len(post_event) < 3:
                continue
            
            # Calculate price impact
            pre_return = pre_event['close'].pct_change().mean() * 100
            post_return = post_event['close'].pct_change().mean() * 100
            
            # Calculate volume impact
            pre_volume_avg = pre_event['volume'].mean()
            post_volume_avg = post_event['volume'].mean()
            volume_change_pct = ((post_volume_avg - pre_volume_avg) / pre_volume_avg) * 100
            
            # Calculate volatility impact
            pre_volatility = pre_event['close'].pct_change().std() * 100
            post_volatility = post_event['close'].pct_change().std() * 100
            volatility_change_pct = ((post_volatility - pre_volatility) / pre_volatility) * 100
            
            # Technical indicator changes
            indicator_changes = {}
            
            # Check key technical indicators if available
            for indicator in ['rsi', 'adx', 'obv', 'cmf', 'atr', 'bollinger_width']:
                if indicator in event_window.columns:
                    pre_indicator_avg = pre_event[indicator].mean()
                    post_indicator_avg = post_event[indicator].mean()
                    
                    if pre_indicator_avg != 0:
                        change_pct = ((post_indicator_avg - pre_indicator_avg) / abs(pre_indicator_avg)) * 100
                        indicator_changes[indicator] = {
                            "pre_avg": round(pre_indicator_avg, 4),
                            "post_avg": round(post_indicator_avg, 4),
                            "change_pct": round(change_pct, 2)
                        }
            
            # Action-specific analysis
            action_details = {}
            if action_type == 'dividend':
                # Calculate dividend yield
                dividend_amount = action['amount']
                price_on_day = price_df.loc[action_date]['close'] if action_date in price_df.index else None
                
                if price_on_day:
                    dividend_yield = (dividend_amount / price_on_day) * 100
                    action_details = {
                        "dividend_amount": dividend_amount,
                        "dividend_yield_pct": round(dividend_yield, 4),
                        "price_on_day": price_on_day
                    }
            
            elif action_type == 'split':
                # Additional details for splits
                action_details = {
                    "split_ratio": action['ratio'],
                    "price_before": price_df.loc[action_date - pd.Timedelta(days=1)]['close'] 
                        if (action_date - pd.Timedelta(days=1)) in price_df.index else None,
                    "price_after": price_df.loc[action_date]['close'] 
                        if action_date in price_df.index else None
                }
            
            # Compile all analysis for this action
            action_impact = {
                "date": action_date.strftime('%Y-%m-%d'),
                "action_type": action_type,
                "price_impact": {
                    "pre_return_pct_avg": round(pre_return, 4),
                    "post_return_pct_avg": round(post_return, 4),
                    "return_change": round(post_return - pre_return, 4)
                },
                "volume_impact": {
                    "pre_volume_avg": int(pre_volume_avg),
                    "post_volume_avg": int(post_volume_avg),
                    "volume_change_pct": round(volume_change_pct, 2)
                },
                "volatility_impact": {
                    "pre_volatility": round(pre_volatility, 4),
                    "post_volatility": round(post_volatility, 4),
                    "volatility_change_pct": round(volatility_change_pct, 2)
                },
                "technical_indicator_changes": indicator_changes,
                "action_details": action_details
            }
            
            impact_results["actions"].append(action_impact)
        
        # Add overall summary if we have action impacts
        if impact_results["actions"]:
            # Calculate averages across all actions
            avg_price_impact = np.mean([a["price_impact"]["return_change"] for a in impact_results["actions"]])
            avg_volume_change = np.mean([a["volume_impact"]["volume_change_pct"] for a in impact_results["actions"]])
            avg_volatility_change = np.mean([a["volatility_impact"]["volatility_change_pct"] for a in impact_results["actions"]])
            
            impact_results["summary"] = {
                "avg_price_impact": round(avg_price_impact, 4),
                "avg_volume_change_pct": round(avg_volume_change, 2),
                "avg_volatility_change_pct": round(avg_volatility_change, 2),
                "impact_assessment": self._assess_impact(avg_price_impact, avg_volume_change, avg_volatility_change)
            }
        
        return impact_results
    
    def _assess_impact(self, price_impact: float, volume_change: float, volatility_change: float) -> str:
        """Generate a qualitative assessment of corporate action impact"""
        assessment = []
        
        # Price impact assessment
        if price_impact > 3:
            assessment.append("Corporate actions have a strongly positive effect on returns")
        elif price_impact > 1:
            assessment.append("Corporate actions have a moderately positive effect on returns")
        elif price_impact < -3:
            assessment.append("Corporate actions have a strongly negative effect on returns")
        elif price_impact < -1:
            assessment.append("Corporate actions have a moderately negative effect on returns")
        else:
            assessment.append("Corporate actions have minimal effect on price returns")
        
        # Volume impact assessment
        if volume_change > 50:
            assessment.append("with dramatically increased trading volume")
        elif volume_change > 20:
            assessment.append("with significantly increased trading volume")
        elif volume_change < -20:
            assessment.append("with significantly decreased trading volume")
        
        # Volatility impact
        if volatility_change > 30:
            assessment.append("and substantially higher price volatility afterward")
        elif volatility_change > 15:
            assessment.append("and moderately higher price volatility afterward")
        elif volatility_change < -15:
            assessment.append("and lower price volatility afterward")
        
        return " ".join(assessment)
    
    def generate_visualization(self, symbol: str, price_df: pd.DataFrame, actions_df: pd.DataFrame) -> Optional[str]:
        """
        Generate visualization of price with corporate actions
        
        Args:
            symbol: Stock symbol
            price_df: DataFrame with price history
            actions_df: DataFrame with corporate actions
            
        Returns:
            Base64 encoded PNG image or None if visualization failed
        """
        if price_df.empty:
            return None
            
        try:
            # Filter actions for this symbol
            symbol_actions = actions_df[actions_df['symbol'] == symbol]
            
            # Convert dates
            if not pd.api.types.is_datetime64_dtype(price_df.index):
                price_df.index = pd.to_datetime(price_df.index)
                
            if not symbol_actions.empty and not pd.api.types.is_datetime64_dtype(symbol_actions['date']):
                symbol_actions['date'] = pd.to_datetime(symbol_actions['date'])
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Plot 1: Price with corporate actions
            ax1.set_title(f"{symbol} Stock Price with Corporate Actions")
            ax1.plot(price_df.index, price_df['close'], label='Close Price')
            
            # Add EMA lines if available
            if 'ema_20' in price_df.columns:
                ax1.plot(price_df.index, price_df['ema_20'], label='EMA 20', linestyle='--', alpha=0.7)
            
            if 'sma_50' in price_df.columns:
                ax1.plot(price_df.index, price_df['sma_50'], label='SMA 50', linestyle='--', alpha=0.7)
                
            if 'sma_200' in price_df.columns:
                ax1.plot(price_df.index, price_df['sma_200'], label='SMA 200', linestyle='--', alpha=0.7)
            
            # Add Bollinger Bands if available
            if all(col in price_df.columns for col in ['bollinger_hband', 'bollinger_lband']):
                ax1.plot(price_df.index, price_df['bollinger_hband'], 'r--', alpha=0.3)
                ax1.plot(price_df.index, price_df['bollinger_lband'], 'r--', alpha=0.3)
                ax1.fill_between(price_df.index, price_df['bollinger_hband'], price_df['bollinger_lband'], 
                                 color='red', alpha=0.1, label='Bollinger Bands')
            
            # Mark corporate actions
            for action_type, marker, color in [
                ('dividend', 'o', 'green'),
                ('split', '^', 'red'),
                ('merger', 's', 'purple')
            ]:
                if not symbol_actions.empty:
                    action_dates = symbol_actions[symbol_actions['action_type'] == action_type]['
                                                                                                # Mark corporate actions
            for action_type, marker, color in [
                ('dividend', 'o', 'green'),
                ('split', '^', 'red'),
                ('merger', 's', 'purple')
            ]:
                if not symbol_actions.empty:
                    action_dates = symbol_actions[symbol_actions['action_type'] == action_type]['date']
                    for date in action_dates:
                        if date in price_df.index:
                            price_at_date = price_df.loc[date, 'close']
                            ax1.scatter(date, price_at_date, marker=marker, color=color, s=100, 
                                       label=f"{action_type.capitalize()}" if date == action_dates.iloc[0] else "")
            
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Volume and OBV
            ax2.bar(price_df.index, price_df['volume'], alpha=0.5, color='blue', label='Volume')
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)
            
            # Add OBV on secondary y-axis if available
            if 'obv' in price_df.columns:
                ax2_2 = ax2.twinx()
                ax2_2.plot(price_df.index, price_df['obv'], color='orange', label='OBV')
                ax2_2.set_ylabel('OBV')
                
                # Combine legends
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                ax2.legend(loc='upper left')
            
            # Plot 3: Technical Indicators
            # Choose indicators based on availability
            available_indicators = []
            for indicator in ['rsi', 'adx', 'cmf', 'williams_r']:
                if indicator in price_df.columns and not price_df[indicator].isna().all():
                    available_indicators.append(indicator)
            
            if available_indicators:
                # Use first available indicator
                indicator = available_indicators[0]
                
                # Plot the indicator
                ax3.plot(price_df.index, price_df[indicator], label=indicator.upper())
                
                # Add horizontal lines for indicators with standard thresholds
                if indicator == 'rsi':
                    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.3)
                    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.3)
                elif indicator == 'williams_r':
                    ax3.axhline(y=-20, color='r', linestyle='--', alpha=0.3)
                    ax3.axhline(y=-80, color='g', linestyle='--', alpha=0.3)
                
                # Add a second indicator if available
                if len(available_indicators) > 1:
                    indicator2 = available_indicators[1]
                    ax3_2 = ax3.twinx()
                    ax3_2.plot(price_df.index, price_df[indicator2], color='green', label=indicator2.upper())
                    
                    # Combine legends
                    lines1, labels1 = ax3.get_legend_handles_labels()
                    lines2, labels2 = ax3_2.get_legend_handles_labels()
                    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                else:
                    ax3.legend(loc='upper left')
            
            ax3.set_ylabel('Indicator Value')
            ax3.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.xticks(rotation=45)
            fig.tight_layout()
            
            # Save figure to buffer
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150)
            buffer.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def parallel_collect(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Collect all corporate actions and technical data for multiple symbols in parallel
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Tuple of (actions_df, technical_data_dict)
        """
        all_actions = []
        technical_data = {}
        impact_analysis = {}
        
        # Define collection function for a single symbol
        def collect_symbol_data(symbol):
            try:
                logger.info(f"Collecting data for {symbol}")
                
                # Get price history for technical analysis
                price_df = self.get_price_history(symbol, start_date, end_date)
                
                # Calculate technical indicators
                if not price_df.empty:
                    price_with_indicators = self.analyze_technical_indicators(price_df)
                    technical_data[symbol] = price_with_indicators
                else:
                    logger.warning(f"No price data found for {symbol}")
                    technical_data[symbol] = pd.DataFrame()
                
                # Get dividends
                dividends_df = self.get_dividends(symbol, start_date, end_date)
                
                # Get splits
                splits_df = self.get_splits(symbol, start_date, end_date)
                
                # Get mergers
                mergers_df = self.get_mergers(symbol, start_date, end_date)
                
                # Combine all actions for this symbol
                symbol_actions = pd.concat([dividends_df, splits_df, mergers_df])
                
                if not symbol_actions.empty:
                    return symbol_actions
                else:
                    return None
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                return None
        
        # Use thread pool to process symbols in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(collect_symbol_data, symbol): symbol for symbol in symbols}
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_actions.append(result)
                except Exception as e:
                    logger.error(f"Exception for {symbol}: {e}")
        
        # Combine all actions
        if all_actions:
            combined_actions = pd.concat(all_actions).reset_index(drop=True)
        else:
            combined_actions = pd.DataFrame(columns=["symbol", "date", "action_type"])
        
        # Analyze impact for each symbol
        for symbol in symbols:
            if symbol in technical_data and not technical_data[symbol].empty:
                impact_analysis[symbol] = self.analyze_corporate_action_impact(
                    symbol, technical_data[symbol], combined_actions
                )
        
        return combined_actions, technical_data, impact_analysis
    
    def summarize(self, actions_df: pd.DataFrame, impact_analysis: Dict) -> Dict:
        """
        Create an intelligent summary of corporate actions and their market impact
        
        Returns a dictionary with summary statistics and insights
        """
        if actions_df.empty:
            return {"error": "No corporate actions found"}
        
        # Convert date column to datetime
        if 'date' in actions_df.columns and not pd.api.types.is_datetime64_dtype(actions_df['date']):
            actions_df['date'] = pd.to_datetime(actions_df['date'])
        
        # Group by symbol and action type
        by_symbol = actions_df.groupby('symbol')
        by_type = actions_df.groupby('action_type')
        
        # Summary statistics by action type
        summary = {
            "total_actions": len(actions_df),
            "symbols_covered": len(actions_df['symbol'].unique()),
            "date_range": {
                "start": actions_df['date'].min().strftime('%Y-%m-%d') if not actions_df.empty else None,
                "end": actions_df['date'].max().strftime('%Y-%m-%d') if not actions_df.empty else None,
            },
            "action_counts": dict(actions_df['action_type'].value_counts()),
            "by_action_type": {},
            "market_impact": {}
        }
        
        # Detailed analysis by action type
        if 'dividend' in by_type.groups:
            dividend_df = by_type.get_group('dividend')
            summary["by_action_type"]["dividend"] = {
                "count": len(dividend_df),
                "symbols": len(dividend_df['symbol'].unique()),
                "avg_amount": round(dividend_df['amount'].mean(), 4) if 'amount' in dividend_df.columns else None,
                "max_amount": round(dividend_df['amount'].max(), 4) if 'amount' in dividend_df.columns else None,
                "top_dividend_symbols": dividend_df.groupby('symbol')['amount'].sum().nlargest(5).to_dict() 
                    if 'amount' in dividend_df.columns else {},
                "dividend_frequency": self._analyze_dividend_frequency(dividend_df)
            }
        
        if 'split' in by_type.groups:
            split_df = by_type.get_group('split')
            summary["by_action_type"]["split"] = {
                "count": len(split_df),
                "symbols": len(split_df['symbol'].unique()),
                "split_types": {
                    "forward_splits": len(split_df[split_df['factor'] > 1]) if 'factor' in split_df.columns else 0,
                    "reverse_splits": len(split_df[split_df['factor'] < 1]) if 'factor' in split_df.columns else 0
                },
                "most_common_ratios": split_df['ratio'].value_counts().head(3).to_dict() if 'ratio' in split_df.columns else {}
            }
        
        if 'merger' in by_type.groups:
            merger_df = by_type.get_group('merger')
            summary["by_action_type"]["merger"] = {
                "count": len(merger_df),
                "symbols": len(merger_df['symbol'].unique()),
                "total_value": merger_df['deal_value'].sum() if 'deal_value' in merger_df.columns else None,
                "avg_deal_value": merger_df['deal_value'].mean() if 'deal_value' in merger_df.columns else None
            }
        
        # Summarize market impact analysis
        if impact_analysis:
            # Collect symbols with substantial data
            valid_symbols = [symbol for symbol, data in impact_analysis.items() 
                            if "actions" in data and len(data["actions"]) > 0]
            
            if valid_symbols:
                # Overall impact statistics
                all_price_impacts = []
                all_volume_impacts = []
                all_volatility_impacts = []
                
                for symbol in valid_symbols:
                    if "summary" in impact_analysis[symbol]:
                        if "avg_price_impact" in impact_analysis[symbol]["summary"]:
                            all_price_impacts.append(impact_analysis[symbol]["summary"]["avg_price_impact"])
                        if "avg_volume_change_pct" in impact_analysis[symbol]["summary"]:
                            all_volume_impacts.append(impact_analysis[symbol]["summary"]["avg_volume_change_pct"])
                        if "avg_volatility_change_pct" in impact_analysis[symbol]["summary"]:
                            all_volatility_impacts.append(impact_analysis[symbol]["summary"]["avg_volatility_change_pct"])
                
                if all_price_impacts:
                    summary["market_impact"]["avg_price_impact"] = round(np.mean(all_price_impacts), 4)
                if all_volume_impacts:
                    summary["market_impact"]["avg_volume_change_pct"] = round(np.mean(all_volume_impacts), 2)
                if all_volatility_impacts:
                    summary["market_impact"]["avg_volatility_change_pct"] = round(np.mean(all_volatility_impacts), 2)
                
                # Dividend specific impacts
                dividend_price_impacts = []
                split_price_impacts = []
                
                for symbol in valid_symbols:
                    for action in impact_analysis[symbol].get("actions", []):
                        if action["action_type"] == "dividend":
                            dividend_price_impacts.append(action["price_impact"]["return_change"])
                        elif action["action_type"] == "split":
                            split_price_impacts.append(action["price_impact"]["return_change"])
                
                if dividend_price_impacts:
                    summary["market_impact"]["dividend_avg_price_impact"] = round(np.mean(dividend_price_impacts), 4)
                if split_price_impacts:
                    summary["market_impact"]["split_avg_price_impact"] = round(np.mean(split_price_impacts), 4)
                
                # Technical indicator impacts
                # Find which indicators are commonly available across actions
                available_indicators = set()
                for symbol in valid_symbols:
                    for action in impact_analysis[symbol].get("actions", []):
                        available_indicators.update(action.get("technical_indicator_changes", {}).keys())
                
                # Collect impacts for each indicator
                indicator_impacts = {indicator: [] for indicator in available_indicators}
                
                for symbol in valid_symbols:
                    for action in impact_analysis[symbol].get("actions", []):
                        for indicator, data in action.get("technical_indicator_changes", {}).items():
                            if "change_pct" in data:
                                indicator_impacts[indicator].append(data["change_pct"])
                
                # Calculate average impact per indicator
                for indicator, impacts in indicator_impacts.items():
                    if impacts:
                        summary["market_impact"][f"{indicator}_avg_change"] = round(np.mean(impacts), 2)
            
            # Add most impactful symbols
            symbol_impacts = {}
            for symbol in valid_symbols:
                if "summary" in impact_analysis[symbol] and "avg_price_impact" in impact_analysis[symbol]["summary"]:
                    symbol_impacts[symbol] = abs(impact_analysis[symbol]["summary"]["avg_price_impact"])
            
            if symbol_impacts:
                top_impact_symbols = sorted(symbol_impacts.items(), key=lambda x: x[1], reverse=True)[:5]
                summary["market_impact"]["most_impactful_symbols"] = {symbol: impact for symbol, impact in top_impact_symbols}
        
        # Generate simple insights
        summary["insights"] = self._generate_enhanced_insights(actions_df, impact_analysis)
        
        return summary
    
    def _analyze_dividend_frequency(self, dividend_df: pd.DataFrame) -> Dict:
        """Analyze frequency patterns in dividend payments"""
        result = {}
        
        # Group by symbol
        for symbol, group in dividend_df.groupby('symbol'):
            if len(group) < 2:
                continue
            
            # Sort by date
            group = group.sort_values('date')
            
            # Calculate days between dividends
            group['days_since_prev'] = group['date'].diff().dt.days
            
            # Skip first row with NaN
            intervals = group['days_since_prev'].dropna().values
            
            if len(intervals) < 2:
                continue
                
            # Calculate average interval and standard deviation
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Determine frequency pattern
            if 80 <= avg_interval <= 100 and std_interval < 15:
                frequency = "quarterly"
            elif 30 <= avg_interval <= 40 and std_interval < 10:
                frequency = "monthly"
            elif 180 <= avg_interval <= 200 and std_interval < 20:
                frequency = "semi-annual"
            elif 350 <= avg_interval <= 380 and std_interval < 30:
                frequency = "annual"
            else:
                frequency = "irregular"
            
            result[symbol] = {
                "frequency": frequency,
                "avg_interval_days": round(avg_interval, 1),
                "consistency": "high" if std_interval < avg_interval * 0.1 else 
                              "medium" if std_interval < avg_interval * 0.2 else "low"
            }
            
        return result
    
    def _generate_enhanced_insights(self, actions_df: pd.DataFrame, impact_analysis: Dict) -> List[str]:
        """Generate insights from corporate actions data and technical impact analysis"""
        insights = []
        
        # Only proceed if we have data
        if actions_df.empty:
            return ["No corporate actions data available for analysis."]
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(actions_df['date']):
            actions_df['date'] = pd.to_datetime(actions_df['date'])
        
        # Group by month to see seasonal patterns
        actions_df['month'] = actions_df['date'].dt.month
        monthly_actions = actions_df.groupby('month')['action_type'].count()
        
        # Find months with highest activity
        if not monthly_actions.empty:
            peak_month = monthly_actions.idxmax()
            peak_count = monthly_actions.max()
            month_name = datetime(2000, peak_month, 1).strftime('%B')
            
            if peak_count > monthly_actions.mean() * 1.5:
                insights.append(f"{month_name} shows significantly higher corporate action activity with {peak_count} events.")
        
        # Analyze dividend patterns
        if 'dividend' in actions_df['action_type'].values:
            dividend_df = actions_df[actions_df['action_type'] == 'dividend']
            
            # Check for dividend clusters
            dividend_df['year_month'] = dividend_df['date'].dt.to_period('M')
            monthly_dividend_count = dividend_df.groupby('year_month').size()
            
            if len(monthly_dividend_count) > 0:
                max_month = monthly_dividend_count.idxmax()
                max_count = monthly_dividend_count.max()
                
                if max_count > monthly_dividend_count.mean() * 2:
                    insights.append(f"Dividend concentration detected in {max_month} with {max_count} payments.")
        
        # Analyze split patterns
        if 'split' in actions_df['action_type'].values:
            split_df = actions_df[actions_df['action_type'] == 'split']
            
            if not split_df.empty and 'factor' in split_df.columns:
                # Count reverse splits
                reverse_splits = split_df[split_df['factor'] < 1]
                if len(reverse_splits) > 0:
                    reverse_pct = len(reverse_splits) / len(split_df) * 100
                    if reverse_pct > 30:
                        insights.append(f"High proportion of reverse splits ({reverse_pct:.1f}%) may indicate distressed sectors.")
        
        # Technical impact insights
        if impact_analysis:
            valid_symbols = [symbol for symbol, data in impact_analysis.items() 
                           if "actions" in data and len(data["actions"]) > 0]
            
            if valid_symbols:
                # Collect impact data
                dividend_price_impacts = []
                split_price_impacts = []
                dividend_volume_impacts = []
                split_volume_impacts = []
                
                for symbol in valid_symbols:
                    for action in impact_analysis[symbol].get("actions", []):
                        if action["action_type"] == "dividend":
                            dividend_price_impacts.append(action["price_impact"]["return_change"])
                            dividend_volume_impacts.append(action["volume_impact"]["volume_change_pct"])
                        elif action["action_type"] == "split":
                            split_price_impacts.append(action["price_impact"]["return_change"])
                            split_volume_impacts.append(action["volume_impact"]["volume_change_pct"])
                
                # Dividend impact insights
                if dividend_price_impacts:
                    avg_dividend_impact = np.mean(dividend_price_impacts)
                    if abs(avg_dividend_impact) > 2:
                        direction = "positive" if avg_dividend_impact > 0 else "negative"
                        insights.append(f"Dividends have a {direction} price impact of {avg_dividend_impact:.2f}% on average.")
                
                if dividend_volume_impacts:
                    avg_dividend_volume = np.mean(dividend_volume_impacts)
                    if abs(avg_dividend_volume) > 25:
                        direction = "increase" if avg_dividend_volume > 0 else "decrease"
                        insights.append(f"Trading volume tends to {direction} by {abs(avg_dividend_volume):.1f}% after dividend announcements.")
                
                # Split impact insights
                if split_price_impacts:
                    avg_split_impact = np.mean(split_price_impacts)
                    if abs(avg_split_impact) > 3:
                        direction = "positive" if avg_split_impact > 0 else "negative"
                        insights.append(f"Stock splits have a {direction} price impact of {avg_split_impact:.2f}% on average.")
                
                # Look for technical indicator patterns
                indicator_insights = []
                for symbol in valid_symbols:
                    for action in impact_analysis[symbol].get("actions", []):
                        for indicator, data in action.get("technical_indicator_changes", {}).items():
                            if "change_pct" in data and abs(data["change_pct"]) > 30:
                                if indicator == "rsi" and data["change_pct"] > 30:
                                    indicator_insights.append(f"RSI typically increases significantly after corporate actions, suggesting momentum shifts.")
                                elif indicator == "bollinger_width" and data["change_pct"] > 40:
                                    indicator_insights.append(f"Volatility (Bollinger Band Width) tends to expand substantially following corporate actions.")
                                elif indicator == "adx" and data["change_pct"] > 25:
                                    indicator_insights.append(f"Trend strength (ADX) typically increases after corporate actions, indicating stronger price movements.")
                                elif indicator == "obv" and data["change_pct"] > 30:
                                    indicator_insights.append(f"On-Balance Volume tends to increase dramatically after corporate actions, suggesting institutional accumulation.")
                
                # Add unique indicator insights
                for insight in list(set(indicator_insights))[:2]:  # Limit to top 2 unique insights
                    insights.append(insight)
        
        # Generate cross-asset class insights if we have multiple symbols
        if len(actions_df['symbol'].unique()) > 5:
            # Group by sector/industry if possible
            # For this example, we'll just group by symbol first letter as a placeholder
            actions_df['sector'] = actions_df['symbol'].str[0]
            sector_counts = actions_df.groupby('sector')['action_type'].count()
            
            if not sector_counts.empty:
                top_sector = sector_counts.idxmax()
                top_count = sector_counts.max()
                if top_count > sector_counts.sum() * 0.4:  # One sector has >40% of all actions
                    insights.append(f"Corporate actions are heavily concentrated in the '{top_sector}' sector, indicating possible industry-wide restructuring or policy changes.")
        
        # Ensure we return at least one insight
        if not insights:
            insights.append("No significant patterns detected in the corporate actions data.")
        
        return insights[:5]  # Limit to top 5 insights for readability
    
    def to_json(self, actions_df: pd.DataFrame, technical_data: Dict, impact_analysis: Dict, summary: Dict, file_path: str) -> None:
        """
        Save corporate actions data, technical analysis, and summary to a JSON file
        
        Args:
            actions_df: DataFrame with corporate actions
            technical_data: Dictionary with technical indicators by symbol
            impact_analysis: Dictionary with impact analysis by symbol
            summary: Dictionary with summary information
            file_path: Path to save the JSON file
        """
        # Convert DataFrame to records for JSON
        actions_list = []
        if not actions_df.empty:
            # Handle datetime conversion for JSON serialization
            df_json = actions_df.copy()
            if 'date' in df_json.columns and pd.api.types.is_datetime64_dtype(df_json['date']):
                df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
            
            actions_list = df_json.to_dict(orient='records')
        
        # Technical indicators summary (avoid including full price history for file size)
        technical_summary = {}
        for symbol, data in technical_data.items():
            if not data.empty:
                # Extract just the latest values of key indicators
                latest_data = data.iloc[-1].to_dict()
                
                # Only include technical indicators, not price/volume data
                indicators = {k: v for k, v in latest_data.items() 
                             if k not in ['open', 'high', 'low', 'close', 'volume', 'adj_close']}
                
                technical_summary[symbol] = {
                    "latest_indicators": indicators,
                    "data_points": len(data),
                    "date_range": {
                        "start": data.index[0].strftime('%Y-%m-%d') if not data.empty else None,
                        "end": data.index[-1].strftime('%Y-%m-%d') if not data.empty else None
                    }
                }
        
        # Complete data structure
        data = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "actions": actions_list,
            "technical_summary": technical_summary,
            "impact_analysis": impact_analysis
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data saved to {file_path}")
    
    def to_excel(self, actions_df: pd.DataFrame, technical_data: Dict, impact_analysis: Dict, summary: Dict, file_path: str) -> None:
        """
        Save corporate actions data and summary to an Excel file with multiple sheets
        
        Args:
            actions_df: DataFrame with corporate actions
            technical_data: Dictionary with technical indicators by symbol
            impact_analysis: Dictionary with impact analysis by symbol
            summary: Dictionary with summary information
            file_path: Path to save the Excel file
        """
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Save all actions to main sheet
            if not actions_df.empty:
                actions_df.to_excel(writer, sheet_name='All Actions', index=False)
                
                # Create separate sheets by action type
                for action_type, group in actions_df.groupby('action_type'):
                    group.to_excel(writer, sheet_name=f'{action_type.capitalize()}s', index=False)
                
                # Create sheet by symbol for top symbols
                for symbol in actions_df['symbol'].value_counts().head(5).index:
                    symbol_df = actions_df[actions_df['symbol'] == symbol]
                    symbol_df.to_excel(writer, sheet_name=f'Symbol {symbol}', index=False)
            
            # Save summary to its own sheet
            summary_rows = []
            
            # Overall statistics
            summary_rows.append(['Total Actions', summary.get('total_actions', 0)])
            summary_rows.append(['Symbols Covered', summary.get('symbols_covered', 0)])
            summary_rows.append(['Date Range', f"{summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}"])
            summary_rows.append([])  # Empty row
            
            # Action type counts
            summary_rows.append(['Action Type', 'Count'])
            for action_type, count in summary.get('action_counts', {}).items():
                summary_rows.append([action_type, count])
            summary_rows.append([])  # Empty row
            
            # Market impact summary
            summary_rows.append(['Market Impact'])
            for impact_type, value in summary.get('market_impact', {}).items():
                if not isinstance(value, dict):  # Skip nested dictionaries
                    summary_rows.append([impact_type.replace('_', ' ').title(), value])
            
            # Insights
            summary_rows.append([])  # Empty row
            summary_rows.append(['Insights'])
            for insight in summary.get('insights', []):
                summary_rows.append([insight])
            
            # Create DataFrame from rows and save
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', header=False, index=False)
            
            # Create Technical Indicators sheet for each symbol with significant data
            for symbol, df in technical_data.items():
                if not df.empty and len(df) > 0:
                    # Extract just the last 30 days for readability
                    last_rows = min(30, len(df))
                    technical_sample = df.tail(last_rows).copy()
                    
                    # Reset index to make date a column
                    technical_sample.reset_index(inplace=True)
                    if 'index' in technical_sample.columns:
                        technical_sample.rename(columns={'index': 'date'}, inplace=True)
                    
                    # Filter columns to include only key indicators
                    key_indicators = ['date', 'close', 'volume', 'rsi', 'adx', 'cci', 'obv', 'cmf', 
                                     'bollinger_width', 'atr', 'stoch_k', 'macd']
                    available_cols = [col for col in key_indicators if col in technical_sample.columns]
                    
                    if len(available_cols) > 1:  # At least date and one other column
                        technical_sample[available_cols].to_excel(writer, sheet_name=f'Tech_{symbol[:7]}', index=False)
            
            # Create Impact Analysis sheet
            impact_rows = []
            impact_rows.append(['Symbol', 'Action Type', 'Date', 'Price Impact', 'Volume Impact', 'Volatility Impact'])
            
            for symbol, analysis in impact_analysis.items():
                for action in analysis.get('actions', []):
                    impact_rows.append([
                        symbol,
                        action.get('action_type', 'N/A'),
                        action.get('date', 'N/A'),
                        action.get('price_impact', {}).get('return_change', 'N/A'),
                        action.get('volume_impact', {}).get('volume_change_pct', 'N/A'),
                        action.get('volatility_impact', {}).get('volatility_change_pct', 'N/A')
                    ])
                    # Continue with Impact Analysis sheet
            if len(impact_rows) > 1:  # Make sure we have data beyond headers
                pd.DataFrame(impact_rows[1:], columns=impact_rows[0]).to_excel(
                    writer, sheet_name='Impact Analysis', index=False)
        
        logger.info(f"Excel report saved to {file_path}")


def main():
    """Example usage of the Corporate Action Collector with advanced technical analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect and analyze corporate actions with technical indicators')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN'], 
                        help='Stock symbols to analyze')
    parser.add_argument('--start-date', default=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'),
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--api-key', help='API key (defaults to FINANCIAL_API_KEY env var)')
    parser.add_argument('--output', default='corporate_actions_analysis.json',
                        help='Output file path')
    parser.add_argument('--excel', action='store_true',
                        help='Also generate Excel report')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate price charts with corporate actions')
    
    args = parser.parse_args()
    
    try:
        # Initialize collector
        collector = CorporateActionCollector(api_key=args.api_key)
        
        # Collect data for all symbols in parallel
        logger.info(f"Collecting corporate actions and technical data for {len(args.symbols)} symbols")
        actions_df, technical_data, impact_analysis = collector.parallel_collect(
            args.symbols, args.start_date, args.end_date
        )
        
        if actions_df.empty:
            logger.warning("No corporate actions found for the specified symbols and date range")
            return
            
        # Generate summary with insights
        logger.info("Generating summary and insights")
        summary = collector.summarize(actions_df, impact_analysis)
        
        # Save to JSON
        collector.to_json(actions_df, technical_data, impact_analysis, summary, args.output)
        
        # Optionally save to Excel
        if args.excel:
            excel_path = args.output.rsplit('.', 1)[0] + '.xlsx'
            collector.to_excel(actions_df, technical_data, impact_analysis, summary, excel_path)
            logger.info(f"Excel report generated: {excel_path}")
        
        # Optionally generate visualizations
        if args.visualize:
            logger.info("Generating visualizations")
            os.makedirs("charts", exist_ok=True)
            
            for symbol in args.symbols:
                if symbol in technical_data and not technical_data[symbol].empty:
                    chart_data = collector.generate_visualization(symbol, technical_data[symbol], actions_df)
                    
                    if chart_data:
                        chart_file = f"charts/{symbol}_with_actions.png"
                        with open(chart_file, "wb") as f:
                            f.write(base64.b64decode(chart_data))
                        logger.info(f"Chart generated: {chart_file}")
        
        logger.info("Corporate actions analysis complete")
        
        # Print key insights to console
        print("\nCorporate Actions Analysis Summary:")
        print(f"- Total actions found: {summary['total_actions']}")
        print(f"- Symbols analyzed: {summary['symbols_covered']}")
        print(f"- Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        print("\nAction Types:")
        for action_type, count in summary.get('action_counts', {}).items():
            print(f"- {action_type.capitalize()}: {count}")
        
        if summary.get('market_impact'):
            print("\nMarket Impact:")
            if 'avg_price_impact' in summary['market_impact']:
                print(f"- Average price impact: {summary['market_impact']['avg_price_impact']}%")
            if 'avg_volume_change_pct' in summary['market_impact']:
                print(f"- Average volume change: {summary['market_impact']['avg_volume_change_pct']}%")
            
        print("\nKey Insights:")
        for insight in summary.get('insights', [])[:5]:
            print(f"- {insight}")
            
        if args.excel:
            print(f"\nDetailed report saved to: {excel_path}")
        print(f"Complete analysis saved to: {args.output}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()