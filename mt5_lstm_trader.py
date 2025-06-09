import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SimpleModelEnhancer:
    def __init__(self, config_path='config.json'):
        logger.info("Initializing Simple Model Enhancer")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # MT5 parameters
        self.mt5_path = self.config['mt5_credentials']['path']
        self.mt5_login = self.config['mt5_credentials']['login']
        self.mt5_password = self.config['mt5_credentials']['password']
        self.mt5_server = self.config['mt5_credentials']['server']

        # Trading parameters
        self.symbols = self.config['symbols'] # Changed to a list of symbols
        self.prediction_timeframe = self.config['prediction_timeframe']
        self.look_back = self.config['look_back']

        # Initialize MT5
        self.init_mt5()

        # Initialize scalers dictionary for multiple symbols
        self.scalers_X = {symbol: MinMaxScaler() for symbol in self.symbols}
        self.scalers_y = {symbol: MinMaxScaler() for symbol in self.symbols}

        # To store trained models (optional, but good for later use)
        self.models = {}

    def init_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize(path=self.mt5_path):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            raise Exception(f"MT5 initialization failed: {mt5.last_error()}")

        if not mt5.login(self.mt5_login, self.mt5_password, self.mt5_server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            raise Exception(f"MT5 login failed: {mt5.last_error()}")

        logger.info(f"Connected to MT5 as {self.mt5_login}")

    def get_historical_data(self, symbol, timeframe, num_bars=1000): # Added symbol argument
        """Get historical data from MT5 for a specific symbol"""
        # Map timeframe string to MT5 constant
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }

        # Handle H6 case
        if timeframe == 'H6':
            logger.info(f"H6 timeframe requested for {symbol}, using H1 data and constructing H6")
            h1_bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, num_bars * 6) # Use passed symbol

            if h1_bars is None or len(h1_bars) == 0:
                logger.error(f"Could not get H1 historical data for {symbol}: {mt5.last_error()}")
                return None

            # Convert to DataFrame
            df_h1 = pd.DataFrame(h1_bars)
            df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')

            # Create a period column to group by every 6 hours
            df_h1['h6_period'] = df_h1['time'].apply(lambda x: x.replace(hour=x.hour - x.hour % 6, minute=0, second=0, microsecond=0))

            # Group by 6-hour periods
            h6_ohlc = df_h1.groupby('h6_period').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'tick_volume': 'sum'
            })

            # Add time as column
            h6_ohlc.reset_index(inplace=True)
            h6_ohlc.rename(columns={'h6_period': 'time'}, inplace=True)

            df = h6_ohlc
        else:
            # Get data for other timeframes
            mt5_timeframe = timeframe_map.get(timeframe)
            if mt5_timeframe is None:
                logger.error(f"Invalid timeframe: {timeframe} for {symbol}")
                return None

            # Get historical data from MT5
            bars = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_bars) # Use passed symbol

            if bars is None or len(bars) == 0:
                logger.error(f"Could not get historical data for {symbol}: {mt5.last_error()}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['time'] = pd.to_datetime(df['time'], unit='s')

        # Ensure all required columns are present
        if 'tick_volume' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['tick_volume']

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'volume' and 'tick_volume' in df.columns:
                    df['volume'] = df['tick_volume']
                else:
                    logger.error(f"Missing required column for {symbol}: {col}")
                    return None

        # Set time as index
        df.set_index('time', inplace=True)

        logger.info(f"Got {len(df)} bars for {symbol} on {timeframe} timeframe")
        return df

    def get_train_test_data(self, symbol): # Added symbol argument
        """Get training and testing data for a specific symbol"""
        # Get historical data
        df_train = self.get_historical_data(symbol, self.prediction_timeframe, 1000) # Pass symbol

        # Get older data for testing (completely separate time period)
        # We'll use a separate time period by first finding the oldest data point
        oldest_date = df_train.index.min()
        test_end_date = oldest_date - timedelta(days=1)  # 1 day buffer
        test_start_date = test_end_date - timedelta(days=100)  # 100 days of test data

        # Convert to timestamp format needed by MT5
        test_end_timestamp = int(test_end_date.timestamp())

        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }

        # Handle H6 timeframe specially for test data
        if self.prediction_timeframe == 'H6':
            # We'll get H1 data and construct H6
            h1_timeframe = mt5.TIMEFRAME_H1
            test_bars = mt5.copy_rates_from(symbol, h1_timeframe, test_end_timestamp, 600) # Use passed symbol

            if test_bars is None or len(test_bars) == 0:
                logger.error(f"Could not get test data for {symbol}: {mt5.last_error()}")
                # Fall back to using part of the training data for testing
                logger.info(f"Falling back to splitting training data for {symbol} for testing")
                return self.split_train_test(df_train)

            # Convert to DataFrame
            df_h1_test = pd.DataFrame(test_bars)
            df_h1_test['time'] = pd.to_datetime(df_h1_test['time'], unit='s')

            # Create a period column to group by every 6 hours
            df_h1_test['h6_period'] = df_h1_test['time'].apply(lambda x: x.replace(hour=x.hour - x.hour % 6, minute=0, second=0, microsecond=0))

            # Group by 6-hour periods
            h6_ohlc_test = df_h1_test.groupby('h6_period').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'tick_volume': 'sum'
            })

            # Add time as column
            h6_ohlc_test.reset_index(inplace=True)
            h6_ohlc_test.rename(columns={'h6_period': 'time'}, inplace=True)
            h6_ohlc_test.set_index('time', inplace=True)

            # Handle columns
            if 'tick_volume' in h6_ohlc_test.columns and 'volume' not in h6_ohlc_test.columns:
                h6_ohlc_test['volume'] = h6_ohlc_test['tick_volume']

            df_test = h6_ohlc_test
        else:
            # For other timeframes, get data directly
            mt5_timeframe = timeframe_map.get(self.prediction_timeframe)
            test_bars = mt5.copy_rates_from(symbol, mt5_timeframe, test_end_timestamp, 200) # Use passed symbol

            if test_bars is None or len(test_bars) == 0:
                logger.error(f"Could not get test data for {symbol}: {mt5.last_error()}")
                # Fall back to using part of the training data for testing
                logger.info(f"Falling back to splitting training data for {symbol} for testing")
                return self.split_train_test(df_train)

            # Convert to DataFrame
            df_test = pd.DataFrame(test_bars)
            df_test['time'] = pd.to_datetime(df_test['time'], unit='s')
            df_test.set_index('time', inplace=True)

            # Handle columns
            if 'tick_volume' in df_test.columns and 'volume' not in df_test.columns:
                df_test['volume'] = df_test['tick_volume']

        logger.info(f"Got {len(df_train)} training samples for {symbol} from {df_train.index.min()} to {df_train.index.max()}")
        logger.info(f"Got {len(df_test)} testing samples for {symbol} from {df_test.index.min()} to {df_test.index.max()}")

        return df_train, df_test

    def split_train_test(self, df):
        """Split data into training and testing sets (fallback method)"""
        # Split data 80/20
        train_size = int(len(df) * 0.8)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]

        logger.warning("Using time-based split instead of truly unseen data")
        return df_train, df_test

    def prepare_data(self, df, symbol, is_training=True): # Added symbol argument
        """Prepare data for LSTM model with added features for a specific symbol"""
        # Add technical indicators
        # 1. Moving Averages
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma14'] = df['close'].rolling(window=14).mean()

        # 2. Price volatility
        df['volatility'] = df['close'].rolling(window=14).std()

        # 3. Price momentum
        df['momentum'] = df['close'] - df['close'].shift(7)

        # 4. High-Low range
        df['range'] = df['high'] - df['low']

        # Drop rows with NaN values
        df = df.dropna()

        # Select features
        feature_cols = ['open', 'high', 'low', 'close', 'volume',
                        'ma7', 'ma14', 'volatility', 'momentum', 'range']
        target_cols = ['high', 'low', 'close']

        features = df[feature_cols].values
        targets = df[target_cols].values

        # Scale features and targets using symbol-specific scalers
        if is_training:
            features_scaled = self.scalers_X[symbol].fit_transform(features)
            targets_scaled = self.scalers_y[symbol].fit_transform(targets)
        else:
            # Use the same scaler fitted on training data
            features_scaled = self.scalers_X[symbol].transform(features)
            targets_scaled = self.scalers_y[symbol].transform(targets)

        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.look_back, len(features_scaled)):
            X.append(features_scaled[i-self.look_back:i])
            y.append(targets_scaled[i])

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build LSTM model with regularization"""
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            80,
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l1_l2(l1=0.0, l2=0.001),
            recurrent_regularizer=l1_l2(l1=0.0, l2=0.001)
        ))
        model.add(Dropout(0.3))

        # Second LSTM layer
        model.add(LSTM(
            60,
            return_sequences=False,
            kernel_regularizer=l1_l2(l1=0.0, l2=0.001)
        ))
        model.add(Dropout(0.3))

        # Output layer
        model.add(Dense(3))

        # Compile model with Adam optimizer and MSE loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )

        return model

    def train_model(self, X_train, y_train, X_val, y_val, symbol): # Added symbol
        """Train the model with validation data for a specific symbol"""
        # Build model
        model = self.build_model(X_train.shape[1:])

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss During Training for {symbol}') # Added symbol
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'training_history_{symbol}.png') # Added symbol to filename
        plt.close()

        # Save the trained model for the current symbol
        model_filename = f'model_{symbol}.keras'
        model.save(model_filename)
        logger.info(f"Model for {symbol} saved to {model_filename}")
        self.models[symbol] = model # Store model

        return model, history

    def evaluate_model(self, model, X, y, data_description="Test", symbol=""): # Added symbol
        """Evaluate model and calculate metrics for a specific symbol"""
        # Get predictions
        y_pred = model.predict(X)

        # Inverse transform predictions and actual values
        y_actual = self.scalers_y[symbol].inverse_transform(y) # Use symbol-specific scaler
        y_pred_inv = self.scalers_y[symbol].inverse_transform(y_pred) # Use symbol-specific scaler

        # Calculate metrics
        mse = mean_squared_error(y_actual, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_pred_inv)
        r2 = r2_score(y_actual.reshape(-1), y_pred_inv.reshape(-1))

        # Calculate direction accuracy (for close price only)
        actual_close = y_actual[:, 2]  # Close is the third column (index 2)
        pred_close = y_pred_inv[:, 2]

        # Calculate direction (up or down)
        actual_direction = np.diff(actual_close) > 0
        pred_direction = np.diff(pred_close) > 0

        # Calculate direction accuracy
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100

        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(actual_close, label='Actual Close', color='blue')
        plt.plot(pred_close, label='Predicted Close', color='red', linestyle='--')
        plt.title(f'{data_description} Data: Actual vs Predicted Close Price for {symbol}') # Added symbol
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{data_description.lower().replace(" ", "_")}_predictions_{symbol}.png') # Added symbol to filename
        plt.close()

        # Print metrics
        logger.info(f"\n{data_description} Metrics for {symbol}:") # Added symbol
        logger.info(f"MSE: {mse:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"R²: {r2:.6f}")
        logger.info(f"Direction Accuracy: {direction_accuracy:.2f}%")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'y_actual': y_actual,
            'y_pred': y_pred_inv
        }

    def perform_cross_validation(self, df, symbol, n_splits=5): # Added symbol
        """Perform time series cross-validation for a specific symbol"""
        # Prepare data
        feature_cols = ['open', 'high', 'low', 'close', 'volume',
                        'ma7', 'ma14', 'volatility', 'momentum', 'range']
        target_cols = ['high', 'low', 'close']

        # Add technical indicators first
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma14'] = df['close'].rolling(window=14).mean()
        df['volatility'] = df['close'].rolling(window=14).std()
        df['momentum'] = df['close'] - df['close'].shift(7)
        df['range'] = df['high'] - df['low']

        # Drop NaN values
        df = df.dropna()

        # Get features and targets
        features = df[feature_cols].values
        targets = df[target_cols].values

        # Scale data using symbol-specific scalers
        features_scaled = self.scalers_X[symbol].fit_transform(features)
        targets_scaled = self.scalers_y[symbol].fit_transform(targets)

        # Create sequences
        X, y = [], []
        for i in range(self.look_back, len(features_scaled)):
            X.append(features_scaled[i-self.look_back:i])
            y.append(targets_scaled[i])

        X = np.array(X)
        y = np.array(y)

        # Create time series cross-validation splits
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"Training fold {fold}/{n_splits} for {symbol}") # Added symbol

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Build and train model
            model = self.build_model(X_train.shape[1:])

            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            ]

            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=50,  # Fewer epochs for CV
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # Evaluate model
            y_pred = model.predict(X_val)

            # Inverse transform predictions and actual values
            y_val_inv = self.scalers_y[symbol].inverse_transform(y_val) # Use symbol-specific scaler
            y_pred_inv = self.scalers_y[symbol].inverse_transform(y_pred) # Use symbol-specific scaler

            # Calculate metrics
            mse = mean_squared_error(y_val_inv, y_pred_inv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_inv, y_pred_inv)
            r2 = r2_score(y_val_inv.reshape(-1), y_pred_inv.reshape(-1))

            # Calculate direction accuracy
            actual_close = y_val_inv[:, 2]
            pred_close = y_pred_inv[:, 2]
            actual_direction = np.diff(actual_close) > 0
            pred_direction = np.diff(pred_close) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100

            # Store metrics
            fold_metrics.append({
                'fold': fold,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            })

            logger.info(f"Fold {fold} for {symbol} - RMSE: {rmse:.6f}, R²: {r2:.6f}, Direction Accuracy: {direction_accuracy:.2f}%") # Added symbol

        # Calculate average metrics
        avg_metrics = {
            'avg_mse': np.mean([m['mse'] for m in fold_metrics]),
            'avg_rmse': np.mean([m['rmse'] for m in fold_metrics]),
            'avg_mae': np.mean([m['mae'] for m in fold_metrics]),
            'avg_r2': np.mean([m['r2'] for m in fold_metrics]),
            'avg_direction_accuracy': np.mean([m['direction_accuracy'] for m in fold_metrics]),
            'std_rmse': np.std([m['rmse'] for m in fold_metrics]),
            'std_direction_accuracy': np.std([m['direction_accuracy'] for m in fold_metrics])
        }

        logger.info(f"\nCross-Validation Results for {symbol}:") # Added symbol
        logger.info(f"Average RMSE: {avg_metrics['avg_rmse']:.6f} ± {avg_metrics['std_rmse']:.6f}")
        logger.info(f"Average R²: {avg_metrics['avg_r2']:.6f}")
        logger.info(f"Average Direction Accuracy: {avg_metrics['avg_direction_accuracy']:.2f}% ± {avg_metrics['std_direction_accuracy']:.2f}%")

        # Plot cross-validation results
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            [m['fold'] for m in fold_metrics],
            [m['direction_accuracy'] for m in fold_metrics],
            yerr=avg_metrics['std_direction_accuracy'],
            fmt='o-',
            capsize=5,
            label='Direction Accuracy'
        )
        plt.axhline(y=avg_metrics['avg_direction_accuracy'], color='r', linestyle='--', alpha=0.7)
        plt.text(1, avg_metrics['avg_direction_accuracy'] + 2, f"Average: {avg_metrics['avg_direction_accuracy']:.2f}%", color='r')
        plt.title(f'Cross-Validation: Direction Accuracy by Fold for {symbol}') # Added symbol
        plt.xlabel('Fold')
        plt.ylabel('Direction Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'cross_validation_results_{symbol}.png') # Added symbol to filename
        plt.close()

        return fold_metrics, avg_metrics

    def run_test(self):
        """Run the complete test for overfitting for all configured symbols"""
        overall_results = {}
        try:
            for symbol in self.symbols: # Loop through all symbols
                logger.info(f"\n{'='*50}\nStarting model enhancement and overfitting test for {symbol}\n{'='*50}")

                # 1. Get train and test data
                df_train, df_test = self.get_train_test_data(symbol) # Pass symbol

                if df_train is None or df_test is None:
                    logger.error(f"Skipping {symbol} due to data fetching issues.")
                    overall_results[symbol] = "skipped"
                    continue

                # 2. Perform cross-validation
                logger.info(f"Performing cross-validation for {symbol}")
                fold_metrics, avg_cv_metrics = self.perform_cross_validation(df_train, symbol) # Pass symbol

                # 3. Prepare training data
                logger.info(f"Preparing training data for {symbol}")
                X_train, y_train = self.prepare_data(df_train, symbol) # Pass symbol

                # 4. Split training data for validation
                val_size = int(len(X_train) * 0.2)
                X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
                y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]

                # 5. Train final model
                logger.info(f"Training final model for {symbol}")
                model, history = self.train_model(X_train_final, y_train_final, X_val, y_val, symbol) # Pass symbol

                # 6. Evaluate on validation data
                logger.info(f"Evaluating on validation data for {symbol}")
                val_metrics = self.evaluate_model(model, X_val, y_val, "Validation", symbol) # Pass symbol

                # 7. Prepare and evaluate on unseen test data
                logger.info(f"Evaluating on unseen test data for {symbol}")
                X_test, y_test = self.prepare_data(df_test, symbol, is_training=False) # Pass symbol
                test_metrics = self.evaluate_model(model, X_test, y_test, "Unseen Test", symbol) # Pass symbol

                # 8. Check for overfitting
                cv_direction = avg_cv_metrics['avg_direction_accuracy']
                test_direction = test_metrics['direction_accuracy']
                direction_drop = cv_direction - test_direction

                cv_rmse = avg_cv_metrics['avg_rmse']
                test_rmse = test_metrics['rmse']
                rmse_increase = ((test_rmse - cv_rmse) / cv_rmse) * 100

                logger.info(f"\nOVERFITTING CHECK for {symbol}:") # Added symbol
                logger.info(f"CV Direction Accuracy: {cv_direction:.2f}%")
                logger.info(f"Test Direction Accuracy: {test_direction:.2f}%")
                logger.info(f"Direction Accuracy Drop: {direction_drop:.2f}%")
                logger.info(f"CV RMSE: {cv_rmse:.6f}")
                logger.info(f"Test RMSE: {test_rmse:.6f}")
                logger.info(f"RMSE Increase: {rmse_increase:.2f}%")

                # Compare test metrics with validation metrics
                val_direction = val_metrics['direction_accuracy']
                val_rmse = val_metrics['rmse']
                val_test_direction_diff = val_direction - test_direction
                val_test_rmse_diff = ((test_rmse - val_rmse) / val_rmse) * 100

                logger.info(f"Validation vs Test Direction Diff: {val_test_direction_diff:.2f}%")
                logger.info(f"Validation vs Test RMSE Diff: {val_test_rmse_diff:.2f}%")

                # Create summary report
                self.create_report(avg_cv_metrics, val_metrics, test_metrics, symbol) # Pass symbol

                # 9. Plot comparison of validation and test predictions
                self.plot_comparison(val_metrics, test_metrics, symbol) # Pass symbol

                # Determine if there's overfitting
                if direction_drop > 10 or rmse_increase > 20 or val_test_direction_diff > 10:
                    logger.warning(f"\nOVERFITTING DETECTED for {symbol}") # Added symbol
                    logger.warning("The model performs significantly worse on unseen test data")
                    logger.warning("Consider increasing regularization, reducing model complexity, or gathering more data")
                    overall_results[symbol] = "overfit"
                else:
                    logger.info(f"\nNO SIGNIFICANT OVERFITTING DETECTED for {symbol}") # Added symbol
                    logger.info("The model generalizes well to unseen data")
                    overall_results[symbol] = "good"

        except Exception as e:
            logger.error(f"Error in model test for a symbol: {e}", exc_info=True)
            overall_results[symbol] = "error" # Store error for the specific symbol
        finally:
            # Cleanup outside the loop, only once
            if mt5.initialize():
                mt5.shutdown()
                logger.info("MT5 connection closed")

        return overall_results # Return results for all symbols

    def create_report(self, cv_metrics, val_metrics, test_metrics, symbol): # Added symbol
        """Create a summary report for a specific symbol"""
        report_filename = f"model_test_report_{symbol}.txt" # Added symbol to filename
        with open(report_filename, "w") as f:
            f.write("=" * 50 + "\n")
            f.write(f"MODEL OVERFITTING TEST REPORT FOR {symbol}\n") # Added symbol
            f.write("=" * 50 + "\n\n")

            f.write("CROSS-VALIDATION METRICS:\n")
            f.write(f"Average RMSE: {cv_metrics['avg_rmse']:.6f} ± {cv_metrics['std_rmse']:.6f}\n")
            f.write(f"Average R²: {cv_metrics['avg_r2']:.6f}\n")
            f.write(f"Average Direction Accuracy: {cv_metrics['avg_direction_accuracy']:.2f}% ± {cv_metrics['std_direction_accuracy']:.2f}%\n\n")

            f.write("VALIDATION METRICS:\n")
            f.write(f"RMSE: {val_metrics['rmse']:.6f}\n")
            f.write(f"R²: {val_metrics['r2']:.6f}\n")
            f.write(f"Direction Accuracy: {val_metrics['direction_accuracy']:.2f}%\n\n")

            f.write("UNSEEN TEST METRICS:\n")
            f.write(f"RMSE: {test_metrics['rmse']:.6f}\n")
            f.write(f"R²: {test_metrics['r2']:.6f}\n")
            f.write(f"Direction Accuracy: {test_metrics['direction_accuracy']:.2f}%\n\n")

            # Compare metrics
            cv_direction = cv_metrics['avg_direction_accuracy']
            test_direction = test_metrics['direction_accuracy']
            direction_drop = cv_direction - test_direction

            cv_rmse = cv_metrics['avg_rmse']
            test_rmse = test_metrics['rmse']
            rmse_increase = ((test_rmse - cv_rmse) / cv_rmse) * 100

            f.write("OVERFITTING ANALYSIS:\n")
            f.write(f"Direction Accuracy Drop (CV to Test): {direction_drop:.2f}%\n")
            f.write(f"RMSE Increase (CV to Test): {rmse_increase:.2f}%\n\n")

            # Determine if there's overfitting
            if direction_drop > 10 or rmse_increase > 20:
                f.write("CONCLUSION: OVERFITTING DETECTED\n")
                f.write("The model performs significantly worse on unseen test data.\n")
                f.write("RECOMMENDATIONS:\n")
                f.write("- Increase dropout rate (currently 0.3)\n")
                f.write("- Increase L2 regularization strength\n")
                f.write("- Reduce model complexity (fewer LSTM units)\n")
                f.write("- Gather more training data\n")
                f.write("- Simplify feature set\n")
            else:
                f.write("CONCLUSION: NO SIGNIFICANT OVERFITTING DETECTED\n")
                f.write("The model generalizes well to unseen data.\n")

        logger.info(f"Report saved to {report_filename}")

    def plot_comparison(self, val_metrics, test_metrics, symbol): # Added symbol
        """Plot comparison of validation and test predictions for a specific symbol"""
        plt.figure(figsize=(12, 8))

        # Plot validation data
        plt.subplot(2, 1, 1)
        plt.plot(val_metrics['y_actual'][:, 2], label='Actual Close', color='blue')
        plt.plot(val_metrics['y_pred'][:, 2], label='Predicted Close', color='red', linestyle='--')
        plt.title(f'Validation Data: Actual vs Predicted Close Price for {symbol}') # Added symbol
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Plot test data
        plt.subplot(2, 1, 2)
        plt.plot(test_metrics['y_actual'][:, 2], label='Actual Close', color='blue')
        plt.plot(test_metrics['y_pred'][:, 2], label='Predicted Close', color='red', linestyle='--')
        plt.title(f'Unseen Test Data: Actual vs Predicted Close Price for {symbol}') # Added symbol
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'validation_vs_test_comparison_{symbol}.png') # Added symbol to filename
        plt.close()

        logger.info(f"Comparison plot saved to validation_vs_test_comparison_{symbol}.png")


if __name__ == "__main__":
    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')

        # Initialize and run test
        enhancer = SimpleModelEnhancer(config_path)
        # The run_test method now returns a dictionary of results per symbol
        results = enhancer.run_test()

        # Determine overall exit code based on all symbol results
        exit_code = 0
        for symbol, status in results.items():
            if status != "good":
                logger.error(f"Test for {symbol} finished with status: {status}")
                exit_code = 1
            else:
                logger.info(f"Test for {symbol} finished with status: {status}")

        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Error during overall model enhancement process: {e}", exc_info=True)
        sys.exit(1)
