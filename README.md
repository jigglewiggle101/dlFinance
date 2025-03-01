# dlFinance

Got it! Here’s an updated version of the `README.md` file that reflects the **MIT License** and includes all the necessary information for users to understand how to use the project, set it up, and test it in a **paper trading environment**.

---

# AI Trading Bot

This project is an **AI-powered trading bot** designed to make intelligent trading decisions using deep learning models such as **LSTM** and **Transformer**. It integrates with **Alpaca API** for live trading (in **paper trading mode** for testing) and uses **Yahoo Finance** for historical data collection.

## Features

- **Data Collection**: Fetch historical stock and cryptocurrency data using Yahoo Finance (`yfinance`) and real-time data from Alpaca API.
- **Model Training**: Train models (LSTM, Transformer) to predict stock prices and make buy/sell/hold decisions.
- **Backtesting**: Simulate trading strategies using historical data to evaluate their performance.
- **Risk Management**: Includes risk management strategies like position sizing, stop-loss, and take-profit mechanisms.
- **Live Trading**: Execute trades on **Alpaca’s paper trading environment** to test the model without using real money.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Backtesting the Strategy](#backtesting-the-strategy)
  - [Live Trading in Paper Trading Mode](#live-trading-in-paper-trading-mode)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Installation

To get started with the AI Trading Bot, follow these steps:

### Setting Up the Virtual Environment

1. **Create a Virtual Environment**:
   Create a Python virtual environment to keep dependencies isolated:
   ```bash
   python -m venv trading-bot-env
   ```

2. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     .\trading-bot-env\Scripts\Activate
     ```
   - On macOS/Linux:
     ```bash
     source trading-bot-env/bin/activate
     ```

### Installing Dependencies

After activating the virtual environment, install the required dependencies using the `requirements.txt` file.

1. **Install Dependencies**:
   Run the following command to install all necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   After installation, confirm that all packages were successfully installed:
   ```bash
   pip show alpaca-trade-api backtrader tensorflow torch scikit-learn pandas yfinance transformers requests SQLAlchemy google-cloud
   ```

---

## Usage

### Training the Model

To train the trading model (e.g., LSTM, Transformer), follow these steps:

1. **Prepare the Data**: 
   - Use the script `data/real_time_data/fetch_data.py` to fetch historical or real-time stock data, or use `yfinance` to collect historical data.

2. **Train the Model**: 
   You can train the model either via Jupyter notebooks or by running the training script. 
   - Open the notebook `notebooks/model_training.ipynb` to interactively train the model. 
   - Alternatively, you can run the script `models/model_training.py` for a non-interactive training process.
   
   In both cases, you can choose between different deep learning models (LSTM or Transformer) and evaluate their performance.

3. **Save the Model**: 
   After training, the trained model will be saved in the `models/` folder, where it can be loaded for testing or live trading.

### Backtesting the Strategy

To test the trading strategy on historical data:

1. **Run Backtesting**:
   - Use `backtesting/backtest.py` or open the Jupyter notebook `notebooks/strategy_testing.ipynb` to backtest the trading strategy using historical stock data.
   - The backtest will simulate trades based on your model’s predictions and evaluate the performance using key metrics like Sharpe ratio, win rate, and profitability.

2. **Evaluate Results**:
   - The backtesting results will be saved in the `backtesting/backtest_results/` folder.
   - Review the performance metrics and adjust the trading strategy accordingly.

### Live Trading in Paper Trading Mode

To test your trading strategy in a **paper trading environment** without risking real money, follow these steps:

1. **Set Up Alpaca API**:
   - Sign up for an **Alpaca account** at [Alpaca Markets](https://alpaca.markets/).
   - Obtain your **API key** and **secret**.
   - Set these values in the `live_trading/config.py` file:
   
   ```python
   API_KEY = "your_api_key"
   API_SECRET = "your_api_secret"
   ```

2. **Run the Live Trading Script**:
   - Run the `live_trading/live_trading.py` script to start trading in paper mode:
   
   ```bash
   python live_trading/live_trading.py
   ```

   - The bot will connect to **Alpaca’s paper trading environment** and execute trades based on the model’s predictions (buy/sell/hold).
   - Monitor the performance in the **Alpaca dashboard** and observe the simulated trades.

3. **Monitoring**:
   - You can set up logging or alerts in `live_trading/live_trading.py` to receive notifications about trade execution and model performance.
   - Optionally, use tools like **Slack** or **email notifications** to receive real-time updates.

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 J

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contributing

We welcome contributions! Please follow these steps to contribute:

1. **Fork the Repository**: 
   Fork this repository to your GitHub account.
   
2. **Create a New Branch**:
   ```bash
   git checkout -b feature-branch
   ```

3. **Make Your Changes**: 
   - Make the necessary changes to the code.
   - Ensure the code follows the project’s coding standards and includes tests.

4. **Commit Your Changes**:
   ```bash
   git commit -am 'Add new feature'
   ```

5. **Push to Your Fork**:
   ```bash
   git push origin feature-branch
   ```

6. **Open a Pull Request**: 
   Create a pull request from your forked repository to the main repository.

---

## Contact

For any inquiries or feedback, feel free to contact:

- **Email**: your.email@example.com
- **GitHub**: [github.com/your-username/AI-Trade-Bot](https://github.com/your-username/AI-Trade-Bot)

---

### **Final Thoughts:**

This **README.md** should give a comprehensive guide on how to use the AI-powered trading bot, train it, backtest strategies, and run it in a **paper trading** environment. Let me know if you need further changes or additions!