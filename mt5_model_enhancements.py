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
        self.symbol = self.config['symbol']
        self.prediction_timeframe = self.config['prediction_timeframe']
        self.look_back = self.config['look_back']
        
        # Initialize MT5
        self.init_mt5()
        
        # Initialize scalers
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
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
    
    def get_historical_data(self, timeframe, num_bars=1000):
        """Get historical data from MT5"""
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
            logger.info("H6 timeframe requested, using H4 data and constructing H6")
            h1_bars = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, num_bars * 6)
            
            if h1_bars is None or len(h1_bars) == 0:
                logger.error(f"Could not get H1 historical data: {mt5.last_error()}")
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
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Get historical data from MT5
            bars = mt5.copy_rates_from_pos(self.symbol, mt5_timeframe, 0, num_bars)
            
            if bars is None or len(bars) == 0:
                logger.error(f"Could not get historical data: {mt5.last_error()}")
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
                    logger.error(f"Missing required column: {col}")
                    return None
        
        # Set time as index
        df.set_index('time', inplace=True)
        
        logger.info(f"Got {len(df)} bars for {timeframe} timeframe")
        return df
    
    def get_train_test_data(self):
        """Get training and testing data"""
        # Get historical data
        df_train = self.get_historical_data(self.prediction_timeframe, 1000)
        
        # Get older data for testing (completely separate time period)
        # We'll use a separate time period by first finding the oldest data point
        oldest_date = df_train.index.min()
        test_end_date = oldest_date - timedelta(days=1)  # 1 day buffer
        test_start_date = test_end_date - timedelta(days=100)  # 100 days of test data
        
        # Convert to timestamp format needed by MT5
        test_end_timestamp = int(test_end_date.timestamp())
        
        # Get historical data for test period
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
            test_bars = mt5.copy_rates_from(self.symbol, h1_timeframe, test_end_timestamp, 600)
            
            if test_bars is None or len(test_bars) == 0:
                logger.error(f"Could not get test data: {mt5.last_error()}")
                # Fall back to using part of the training data for testing
                logger.info("Falling back to splitting training data for testing")
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
            test_bars = mt5.copy_rates_from(self.symbol, mt5_timeframe, test_end_timestamp, 200)
            
            if test_bars is None or len(test_bars) == 0:
                logger.error(f"Could not get test data: {mt5.last_error()}")
                # Fall back to using part of the training data for testing
                logger.info("Falling back to splitting training data for testing")
                return self.split_train_test(df_train)
            
            # Convert to DataFrame
            df_test = pd.DataFrame(test_bars)
            df_test['time'] = pd.to_datetime(df_test['time'], unit='s')
            df_test.set_index('time', inplace=True)
            
            # Handle columns
            if 'tick_volume' in df_test.columns and 'volume' not in df_test.columns:
                df_test['volume'] = df_test['tick_volume']
        
        logger.info(f"Got {len(df_train)} training samples from {df_train.index.min()} to {df_train.index.max()}")
        logger.info(f"Got {len(df_test)} testing samples from {df_test.index.min()} to {df_test.index.max()}")
        
        return df_train, df_test
    
    def split_train_test(self, df):
        """Split data into training and testing sets (fallback method)"""
        # Split data 80/20
        train_size = int(len(df) * 0.8)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
        
        logger.warning("Using time-based split instead of truly unseen data")
        return df_train, df_test
    
    def prepare_data(self, df, is_training=True):
        """Prepare data for LSTM model with added features"""
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
        
        # Scale features and targets
        if is_training:
            features_scaled = self.scaler_X.fit_transform(features)
            targets_scaled = self.scaler_y.fit_transform(targets)
        else:
            # Use the same scaler fitted on training data
            features_scaled = self.scaler_X.transform(features)
            targets_scaled = self.scaler_y.transform(targets)
        
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
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the model with validation data"""
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
        plt.title('Model Loss During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.close()
        
        return model, history
    
    def evaluate_model(self, model, X, y, data_description="Test"):
        """Evaluate model and calculate metrics"""
        # Get predictions
        y_pred = model.predict(X)
        
        # Inverse transform predictions and actual values
        y_actual = self.scaler_y.inverse_transform(y)
        y_pred_inv = self.scaler_y.inverse_transform(y_pred)
        
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
        plt.title(f'{data_description} Data: Actual vs Predicted Close Price')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{data_description.lower()}_predictions.png')
        plt.close()
        
        # Print metrics
        logger.info(f"\n{data_description} Metrics:")
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
    
    def perform_cross_validation(self, df, n_splits=5):
        """Perform time series cross-validation"""
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
        
        # Scale data
        features_scaled = self.scaler_X.fit_transform(features)
        targets_scaled = self.scaler_y.fit_transform(targets)
        
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
            logger.info(f"Training fold {fold}/{n_splits}")
            
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
            y_val_inv = self.scaler_y.inverse_transform(y_val)
            y_pred_inv = self.scaler_y.inverse_transform(y_pred)
            
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
            
            logger.info(f"Fold {fold} - RMSE: {rmse:.6f}, R²: {r2:.6f}, Direction Accuracy: {direction_accuracy:.2f}%")
        
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
        
        logger.info("\nCross-Validation Results:")
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
        plt.title('Cross-Validation: Direction Accuracy by Fold')
        plt.xlabel('Fold')
        plt.ylabel('Direction Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig('cross_validation_results.png')
        plt.close()
        
        return fold_metrics, avg_metrics
    
    def run_test(self):
        """Run the complete test for overfitting"""
        try:
            logger.info("Starting model enhancement and overfitting test")
            
            # 1. Get train and test data
            df_train, df_test = self.get_train_test_data()
            
            # 2. Perform cross-validation
            logger.info("Performing cross-validation")
            fold_metrics, avg_cv_metrics = self.perform_cross_validation(df_train)
            
            # 3. Prepare training data
            logger.info("Preparing training data")
            X_train, y_train = self.prepare_data(df_train)
            
            # 4. Split training data for validation
            val_size = int(len(X_train) * 0.2)
            X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
            y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]
            
            # 5. Train final model
            logger.info("Training final model")
            model, history = self.train_model(X_train_final, y_train_final, X_val, y_val)
            
            # 6. Evaluate on validation data
            logger.info("Evaluating on validation data")
            val_metrics = self.evaluate_model(model, X_val, y_val, "Validation")
            
            # 7. Prepare and evaluate on unseen test data
            logger.info("Evaluating on unseen test data")
            X_test, y_test = self.prepare_data(df_test, is_training=False)
            test_metrics = self.evaluate_model(model, X_test, y_test, "Unseen Test")
            
            # 8. Check for overfitting
            cv_direction = avg_cv_metrics['avg_direction_accuracy']
            test_direction = test_metrics['direction_accuracy']
            direction_drop = cv_direction - test_direction
            
            cv_rmse = avg_cv_metrics['avg_rmse']
            test_rmse = test_metrics['rmse']
            rmse_increase = ((test_rmse - cv_rmse) / cv_rmse) * 100
            
            logger.info("\nOVERFITTING CHECK:")
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
            self.create_report(avg_cv_metrics, val_metrics, test_metrics)
            
            # 9. Plot comparison of validation and test predictions
            self.plot_comparison(val_metrics, test_metrics)
            
            # Determine if there's overfitting
            if direction_drop > 10 or rmse_increase > 20 or val_test_direction_diff > 10:
                logger.warning("\nOVERFITTING DETECTED")
                logger.warning("The model performs significantly worse on unseen test data")
                logger.warning("Consider increasing regularization, reducing model complexity, or gathering more data")
                return "overfit"
            else:
                logger.info("\nNO SIGNIFICANT OVERFITTING DETECTED")
                logger.info("The model generalizes well to unseen data")
                return "good"
        
        except Exception as e:
            logger.error(f"Error in model test: {e}", exc_info=True)
            return "error"
        finally:
            # Clean up
            if mt5.initialize():
                mt5.shutdown()
                logger.info("MT5 connection closed")
    
    def create_report(self, cv_metrics, val_metrics, test_metrics):
        """Create a summary report"""
        with open("model_test_report.txt", "w") as f:
            f.write("=" * 50 + "\n")
            f.write("MODEL OVERFITTING TEST REPORT\n")
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
        
        logger.info(f"Report saved to model_test_report.txt")
    
    def plot_comparison(self, val_metrics, test_metrics):
        """Plot comparison of validation and test predictions"""
        plt.figure(figsize=(12, 8))
        
        # Plot validation data
        plt.subplot(2, 1, 1)
        plt.plot(val_metrics['y_actual'][:, 2], label='Actual Close', color='blue')
        plt.plot(val_metrics['y_pred'][:, 2], label='Predicted Close', color='red', linestyle='--')
        plt.title('Validation Data: Actual vs Predicted Close Price')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot test data
        plt.subplot(2, 1, 2)
        plt.plot(test_metrics['y_actual'][:, 2], label='Actual Close', color='blue')
        plt.plot(test_metrics['y_pred'][:, 2], label='Predicted Close', color='red', linestyle='--')
        plt.title('Unseen Test Data: Actual vs Predicted Close Price')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('validation_vs_test_comparison.png')
        plt.close()
        
        logger.info("Comparison plot saved to validation_vs_test_comparison.png")


if __name__ == "__main__":
    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
        
        # Initialize and run test
        enhancer = SimpleModelEnhancer(config_path)
        result = enhancer.run_test()
        
        sys.exit(0 if result == "good" else 1)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)