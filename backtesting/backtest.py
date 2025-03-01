 # Backtesting framework using Backtrader or other libraries
 
import backtrader as bt
import yfinance as yf

class BuyAndHold(bt.Strategy):
    def __init__(self):
        self.buy_signal = bt.indicators.MACD(self.data)

    def next(self):
        if not self.position:
            self.buy()
        else:
            self.sell()

# Create the backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(BuyAndHold)

# Download historical data (e.g., AAPL stock data)
data = bt.feeds.YahooFinanceData(dataname='AAPL', fromdate='2015-01-01', todate='2020-12-31')
cerebro.adddata(data)

# Run the backtest
cerebro.run()

# Plot the results
cerebro.plot()
