# LSTM-based Multi-Timeframe Trading Bot with Trailing Stop EA

## Overview

This repository contains a comprehensive algorithmic trading solution that combines machine learning prediction with automated trade management. The system consists of two main components:

1. **Python LSTM Trading Bot**: A deep learning model that predicts price movements for forex or other financial instruments using LSTM neural networks.
2. **MQL5 Trailing Stop EA**: A complementary MetaTrader 5 Expert Advisor that manages open positions with advanced trailing stop functionality.

This approach separates prediction (Python) from execution management (MQL5), allowing each component to excel at what it does best.

## Features

### Python LSTM Trading Bot
- **Multi-timeframe Analysis**: Predicts market movements using different timeframes (H1 execution with H4/H6 prediction timeframes)
- **Deep Learning Models**: Implements LSTM neural networks from TensorFlow for time-series prediction
- **Automatic Retraining**: Periodically retrains the model to adapt to changing market conditions
- **Performance Metrics**: Evaluates model with direction accuracy, RMSE, and other key trading metrics
- **Risk Management**: Implements configurable stop-loss and take-profit logic
- **Telegram Integration**: Sends real-time alerts and updates to your mobile device

### MQL5 Trailing Stop EA
- **Flexible Position Management**: Manages open positions with customizable trailing stop logic
- **Multiple Filtering Options**: Can manage all orders or filter by magic number or SL/TP levels
- **Separate Trailing for SL/TP**: Independently trail stop-loss and take-profit levels
- **Configurable Parameters**: Adjust trailing distances and steps to fit your strategy

## System Requirements

- Python 3.6+
- MetaTrader 5 platform
- Telegram account (for notifications)

## Python Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
MetaTrader5
python-telegram-bot
```

## Configuration

The system uses a `config.json` file for all major parameters:

```json
{
    "mt5_credentials": {
        "login": 111111111111,
        "password": "xxxxxxxxxxxxxxx",
        "server": "AdmiralsGroup-Demo",
        "path": "C:\\Program Files\\Admirals Group MT5 Terminal\\terminal64.exe"
    },
    
    "symbol": "EURUSD",
    "prediction_timeframe": "H4",
    "execution_timeframe": "H1",
    "look_back": 120,
    "retraining_hours": 24,
    
    "lot_size": 0.01,
    "tp_multiplier": 2.5,
    "sl_multiplier": 1.0,
    "trailing_start_pct": 0.4,
    "trailing_step_pct": 0.1,
    "risk_per_trade_pct": 1.0,
    
    "telegram_bot_token": "xxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "telegram_chat_id": "1111111111111111111",
    
    "confidence_threshold": 0.75,
    "price_change_threshold": 0.4,
    
    "max_data_points": 50000,
    
    "prediction_validity_hours": 6
}
```

## Installation & Setup

1. Clone this repository
2. Install required Python packages: `pip install -r requirements.txt`
3. Configure your MT5 credentials and other parameters in `config.json`
4. Set up a Telegram bot and add your token and chat ID to the configuration
5. Compile the TrailingEA_003.mq5 in MetaTrader 5
6. Run the bot with `python run-bot.py`

## Usage

### Starting the Python Bot

```bash
python run-bot.py
```

This will:
1. Install any missing dependencies
2. Initialize the MT5 connection
3. Set up the LSTM model and training data
4. Begin the trading loop with the parameters from `config.json`

### Setting up the Trailing Stop EA

1. Attach the TrailingEA_003.mq5 to your chart in MetaTrader 5
2. Configure the EA parameters:
   - Choose the trailing mode (ALL_ORDERS, BY_MAGIC_NUMBER, or BY_SL_TP_LEVELS)
   - Set the trailing distances and steps
   - If using magic number filtering, set the TargetMagicNumber to match your Python bot (default 12345)
   - If using SL/TP filtering, set your desired range parameters

## How It Works

### LSTM Model & Prediction

The Python bot uses a Long Short-Term Memory (LSTM) neural network to predict future price movements based on historical data. The model:

1. Collects OHLCV data from MT5
2. Preprocesses and normalizes the data
3. Trains an LSTM model with regularization to avoid overfitting
4. Predicts future price movements (high, low, close)
5. Evaluates prediction confidence and potential profit

If the predicted price movement exceeds the configured threshold, the bot places a trade with appropriate stop-loss and take-profit levels.

### Trailing Stop Logic

The MQL5 EA runs independently to manage open positions:

1. Monitors all positions matching your filter criteria (magic number or SL/TP levels)
2. Adjusts stop-loss levels as the market moves in your favor
3. Optionally adjusts take-profit levels for maximizing gains
4. Respects configured trailing distances and step sizes

## Model Evaluation & Maintenance

The system includes tools for evaluating and maintaining model performance:

- `mt5_model_enhancements.py`: Tests for and mitigates overfitting issues
- `run_model_evaluation.py`: Evaluates model performance on unseen data
- `logging_fix.py`: Ensures proper logging throughout the system

Regularly review the generated metrics and plots to assess your model's performance.

## Logging

The system maintains comprehensive logs in `trading_bot.log`. For enhanced logging, run `enable_logging.py` before starting the bot.

## Disclaimer

This trading system is provided for educational and research purposes only. Trading financial markets carries significant risk. Always test thoroughly on demo accounts before using real funds.

## License

[MIT License](LICENSE)