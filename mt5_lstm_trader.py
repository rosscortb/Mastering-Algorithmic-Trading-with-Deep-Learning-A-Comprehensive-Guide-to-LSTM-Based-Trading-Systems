import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import MetaTrader5 as mt5
import telegram
import warnings

warnings.filterwarnings('ignore')

# Setup proper logging system
def _setup_logger():
    logger = logging.getLogger(__name__)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure logger
    logger.setLevel(logging.INFO)
    
    # Only add handlers if the root logger doesn't have any yet
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler('trading_bot.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)
    
    return logger

logger = _setup_logger()

class MT5LSTMTrader:
    def __init__(self, config_path='config.json'):
        # Guardar el directorio de trabajo
        self.script_dir = os.path.dirname(os.path.abspath(config_path))
        logger.info(f"Directorio de trabajo: {self.script_dir}")
        
        # Cargar configuración
        self.load_config(config_path)
        
        # Add this line to fix the error - use execution_timeframe as the default timeframe
        self.timeframe = self.execution_timeframe
        
        # Inicializar MT5
        self.init_mt5()
        
        # Inicializar Telegram Bot
        self.init_telegram()
        
        # Inicializar escaladores
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        # Variables del modelo
        self.model = None
        self.model_path = os.path.join(self.script_dir, 'lstm_model.h5')
        self.last_trained = None
        
        # Métricas y resultados
        self.metrics = {
            'mse': [],
            'mae': [],
            'r2': [],
            'rmse': [],
            'mape': [],
            'correlation': [],
            'direction_accuracy': [],
            'sharpe_ratio': []
        }
        
        # Variables para control de órdenes
        self.last_candle_time = None
        self.order_placed_for_current_candle = False
        
        # Variables para la estrategia multi-timeframe
        self.current_prediction = None
        self.prediction_time = None
        self.last_h6_candle_time = None

    def _calculate_additional_metrics(self, y_true, y_pred):
        """Calcula métricas adicionales para evaluar el rendimiento del modelo"""
        try:
            import numpy as np
            from sklearn.metrics import mean_absolute_percentage_error
            
            # Convertir arrays a formato plano si es necesario
            if len(y_true.shape) > 1:
                y_true_flat = y_true.flatten()
                y_pred_flat = y_pred.flatten()
            else:
                y_true_flat = y_true
                y_pred_flat = y_pred
            
            # Calcular métricas comunes
            mae = mean_absolute_error(y_true_flat, y_pred_flat)
            mse = mean_squared_error(y_true_flat, y_pred_flat)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_flat, y_pred_flat)
            
            # Calcular error porcentual absoluto medio (MAPE)
            # Filtrar para evitar divisiones por cero
            mask = y_true_flat != 0
            if np.any(mask):
                mape = mean_absolute_percentage_error(y_true_flat[mask], y_pred_flat[mask]) * 100
            else:
                mape = np.nan
                
            # Calcular el coeficiente de correlación
            corr = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
            
            # Calcular la dirección correcta de predicción
            # (Este es un indicador muy importante para trading)
            direction_true = np.diff(y_true_flat) > 0
            direction_pred = np.diff(y_pred_flat) > 0
            direction_accuracy = np.mean(direction_true == direction_pred) * 100 if len(direction_true) > 0 else np.nan
            
            # Calcular métricas específicas para trading
            # Relación de Sharpe simple (basada en la dirección)
            # Asumimos que cada predicción correcta de dirección es un "retorno" positivo
            returns = (direction_true == direction_pred) * 2 - 1  # Convierte True/False a 1/-1
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else np.nan
            
            # Crear diccionario de métricas
            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'correlation': corr,
                'direction_accuracy': direction_accuracy,
                'sharpe_ratio': sharpe_ratio
            }
            
            # Logear las métricas
            logger.info(f"Métricas adicionales calculadas - RMSE: {rmse:.5f}, MAPE: {mape:.2f}%, "
                    f"Dirección: {direction_accuracy:.2f}%, Sharpe: {sharpe_ratio:.2f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error al calcular métricas adicionales: {e}")
            return {
                'mae': np.nan, 
                'mse': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'correlation': np.nan,
                'direction_accuracy': np.nan,
                'sharpe_ratio': np.nan
            }
        

    def is_prediction_valid(self):
        """Verifica si la predicción actual sigue siendo válida"""
        if self.prediction_time is None or self.current_prediction is None:
            logger.info("No hay predicción actual, se requiere una nueva")
            return False
            
        # Verificar si ha pasado el tiempo de validez desde la última predicción
        time_since_prediction = datetime.now() - self.prediction_time
        is_valid = time_since_prediction.total_seconds() < (self.prediction_validity_hours * 3600)
        
        if not is_valid:
            logger.info(f"Predicción expirada, han pasado {time_since_prediction.total_seconds()/3600:.2f} horas")
        else:
            logger.info(f"Predicción todavía válida, generada hace {time_since_prediction.total_seconds()/3600:.2f} horas")
            
        return is_valid
    
    def evaluate_model_performance(self, y_test, y_pred):
        """Evalúa y registra el rendimiento del modelo con métricas adicionales"""
        try:
            # Calcular métricas básicas
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test.flatten(), y_pred.flatten())
            
            # Guardar métricas básicas
            self.metrics['mse'].append(mse)
            self.metrics['mae'].append(mae)
            self.metrics['r2'].append(r2)
            
            # Calcular métricas adicionales
            additional_metrics = self._calculate_additional_metrics(y_test, y_pred)
            
            # Guardar métricas adicionales si no existen las listas
            for key, value in additional_metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
            
            # Crear informe de métricas para el registro
            report = (
                f"Evaluación del Modelo:\n"
                f"MSE: {mse:.5f}\n"
                f"RMSE: {additional_metrics['rmse']:.5f}\n"
                f"MAE: {mae:.5f}\n"
                f"MAPE: {additional_metrics['mape']:.2f}%\n"
                f"R²: {r2:.5f}\n"
                f"Correlación: {additional_metrics['correlation']:.5f}\n"
                f"Precisión de Dirección: {additional_metrics['direction_accuracy']:.2f}%\n"
                f"Sharpe Ratio: {additional_metrics['sharpe_ratio']:.2f}\n"
            )
            
            logger.info(report)
            return additional_metrics
            
        except Exception as e:
            logger.error(f"Error al evaluar el rendimiento del modelo: {e}")
            return None

    def _get_wait_time(self, timeframe):
        """Calcula el tiempo de espera optimizado para capturar el inicio de nuevas velas"""
        try:
            # Obtener tiempo actual
            current_time = datetime.now()
            
            # Determinar el tiempo del próximo inicio de vela basado en el timeframe
            if timeframe == "M1":
                # Para velas de 1 minuto, siguiente minuto exacto
                next_candle = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
            elif timeframe == "M5":
                # Para velas de 5 minutos
                minutes_to_add = 5 - (current_time.minute % 5)
                if minutes_to_add == 0:
                    minutes_to_add = 5
                next_candle = current_time.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
            elif timeframe == "M15":
                # Para velas de 15 minutos
                minutes_to_add = 15 - (current_time.minute % 15)
                if minutes_to_add == 0:
                    minutes_to_add = 15
                next_candle = current_time.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
            elif timeframe == "M30":
                # Para velas de 30 minutos
                minutes_to_add = 30 - (current_time.minute % 30)
                if minutes_to_add == 0:
                    minutes_to_add = 30
                next_candle = current_time.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
            elif timeframe == "H1":
                # Para velas de 1 hora, siguiente hora exacta
                next_candle = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            elif timeframe == "H4":
                # Para velas de 4 horas
                hours_to_add = 4 - (current_time.hour % 4)
                if hours_to_add == 0:
                    hours_to_add = 4
                next_candle = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
            elif timeframe == "H6":
                # Para velas de 6 horas
                hours_to_add = 6 - (current_time.hour % 6)
                if hours_to_add == 0:
                    hours_to_add = 6
                next_candle = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
            elif timeframe == "D1":
                # Para velas diarias, siguiente día a las 00:00
                next_candle = (current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
            else:
                # Timeframe desconocido, verificar cada minuto
                logger.warning(f"Timeframe {timeframe} no reconocido, revisando cada minuto")
                return 60

            # Calcular los segundos hasta la próxima vela
            wait_seconds = (next_candle - current_time).total_seconds()
            
            # Ajustar tiempo de espera para intentar estar ligeramente antes del inicio de la vela
            # Esto asegura que estemos listos para actuar en cuanto se forme la nueva vela
            optimized_wait = max(1, wait_seconds - 2)  # Restar 2 segundos, pero mantener al menos 1 segundo
            
            logger.info(f"Próxima vela {timeframe} a las {next_candle.strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"esperando {optimized_wait:.1f} segundos")
            
            return optimized_wait, next_candle
        
        except Exception as e:
            logger.error(f"Error calculando tiempo de espera: {e}")
            return 60, current_time + timedelta(minutes=1)  # Valor seguro por defecto

    def load_config(self, config_path):
        """Carga la configuración desde el archivo JSON"""
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
                
            # MetaTrader5 credentials
            self.mt5_login = config['mt5_credentials']['login']
            self.mt5_password = config['mt5_credentials']['password']
            self.mt5_server = config['mt5_credentials']['server']
            self.mt5_path = config['mt5_credentials']['path']
            
            # Trading parameters
            self.symbol = config['symbol']
            self.prediction_timeframe = config.get('prediction_timeframe', 'H6')
            self.execution_timeframe = config.get('execution_timeframe', 'H1')
            self.timeframe_dict = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5, 
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'H6': mt5.TIMEFRAME_H6 if hasattr(mt5, 'TIMEFRAME_H6') else mt5.TIMEFRAME_H4*1.5,
                'D1': mt5.TIMEFRAME_D1
            }
            self.look_back = config['look_back']
            self.retraining_hours = config['retraining_hours']
            
            # Comprobar si H6 existe y si no, crear una función para manejarlo
            if not hasattr(mt5, 'TIMEFRAME_H6'):
                logger.warning("MT5 no tiene TIMEFRAME_H6 nativo, se creará una implementación personalizada")
                # Crearemos una función especial para manejar H6
            
            # Order parameters
            self.lot_size = config['lot_size']
            self.tp_multiplier = config['tp_multiplier']
            self.sl_multiplier = config['sl_multiplier']
            self.trailing_start_pct = config['trailing_start_pct']
            self.trailing_step_pct = config['trailing_step_pct']
            self.risk_per_trade_pct = config['risk_per_trade_pct']
            
            # Telegram parameters
            self.telegram_token = config['telegram_bot_token']
            self.telegram_chat_id = config['telegram_chat_id']
            
            # Model parameters
            self.confidence_threshold = config['confidence_threshold']
            self.price_change_threshold = config['price_change_threshold']
            self.max_data_points = config['max_data_points']
            
            # Multi-timeframe strategy parameters
            self.prediction_validity_hours = config.get('prediction_validity_hours', 6)
            
            logger.info(f"Configuración cargada correctamente desde {config_path}")
        except Exception as e:
            logger.error(f"Error al cargar la configuración: {e}")
            raise

    def init_mt5(self):
        """Inicializa la conexión con MetaTrader 5"""
        try:
            if not mt5.initialize(path=self.mt5_path):
                logger.error(f"Error al inicializar MT5: {mt5.last_error()}")
                raise Exception(f"MT5 inicialización fallida: {mt5.last_error()}")
            
            # Login a la cuenta
            if not mt5.login(self.mt5_login, self.mt5_password, self.mt5_server):
                logger.error(f"Error al iniciar sesión en MT5: {mt5.last_error()}")
                mt5.shutdown()
                raise Exception(f"MT5 login fallido: {mt5.last_error()}")
            
            logger.info(f"Conectado a MT5 como {self.mt5_login}")
        except Exception as e:
            logger.error(f"Error en la inicialización de MT5: {e}")
            raise

    def init_telegram(self):
        """Inicializa el bot de Telegram"""
        try:
            self.telegram_bot = telegram.Bot(token=self.telegram_token)
            logger.info("Bot de Telegram inicializado")
            self.send_telegram_message("🤖 Bot de Trading Multi-Timeframe iniciado correctamente!")
        except Exception as e:
            logger.error(f"Error al inicializar Telegram: {e}")
            self.telegram_bot = None

    def send_telegram_message(self, message, image_path=None):
        """Envía un mensaje a Telegram, opcionalmente con una imagen"""
        if not self.telegram_bot:
            logger.warning("Bot de Telegram no inicializado, no se enviará el mensaje")
            return
        
        try:
            self.telegram_bot.send_message(chat_id=self.telegram_chat_id, text=message)
            
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as img:
                    self.telegram_bot.send_photo(chat_id=self.telegram_chat_id, photo=img)
            
            logger.info("Mensaje enviado a Telegram correctamente")
        except Exception as e:
            logger.error(f"Error al enviar mensaje a Telegram: {e}")

    def get_h6_data(self, num_bars=None):
        """Obtiene datos históricos para el timeframe de 6 horas"""
        if num_bars is None:
            num_bars = self.max_data_points
        
        try:
            # Verificar si H6 es un timeframe nativo en MT5
            if hasattr(mt5, 'TIMEFRAME_H6') and isinstance(mt5.TIMEFRAME_H6, int):
                # Usar el timeframe nativo si existe
                bars = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H6, 0, num_bars)
                if bars is not None and len(bars) > 0:
                    df = pd.DataFrame(bars)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    logger.info(f"Obtenidos {len(df)} datos de H6 usando timeframe nativo")
                    return df
            
            # Si no existe H6 nativo o falló, construimos H6 a partir de H1
            # Obtener suficientes datos H1 para construir las velas H6
            h1_bars_needed = num_bars * 6 + 5  # +5 para manejar desalineaciones
            h1_bars = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, h1_bars_needed)
            
            if h1_bars is None or len(h1_bars) == 0:
                logger.error(f"No se pudieron obtener datos históricos H1: {mt5.last_error()}")
                raise Exception(f"Error al obtener datos históricos H1: {mt5.last_error()}")
            
            # Convertir a DataFrame
            df_h1 = pd.DataFrame(h1_bars)
            df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
            
            # Crear una columna de período para agrupar por cada 6 horas
            df_h1['h6_period'] = df_h1['time'].apply(lambda x: x.replace(hour=x.hour - x.hour % 6, minute=0, second=0, microsecond=0))
            
            # Agrupar por periodos de 6 horas
            h6_ohlc = df_h1.groupby('h6_period').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'tick_volume': 'sum' if 'tick_volume' in df_h1.columns else None,
                'volume': 'sum' if 'volume' in df_h1.columns else None,
                'spread': 'mean' if 'spread' in df_h1.columns else None
            })
            
            # Remover columnas que son None (porque no estaban en el dataframe original)
            h6_ohlc = h6_ohlc.dropna(axis=1, how='all')
            
            # Si falta 'volume' pero tenemos 'tick_volume', mapeamos
            if 'tick_volume' in h6_ohlc.columns and 'volume' not in h6_ohlc.columns:
                h6_ohlc['volume'] = h6_ohlc['tick_volume']
                logger.info("Columna 'tick_volume' mapeada a 'volume' en datos H6 construidos")
            
            # Limitar al número de barras solicitado
            h6_ohlc = h6_ohlc.iloc[-num_bars:] if len(h6_ohlc) > num_bars else h6_ohlc
            
            logger.info(f"Construidos {len(h6_ohlc)} datos de H6 a partir de datos H1")
            return h6_ohlc
        
        except Exception as e:
            logger.error(f"Error al obtener datos H6: {e}")
            raise

    def get_historical_data(self, timeframe=None, num_bars=None):
        """Obtiene datos históricos de MT5 para el timeframe especificado"""
        if timeframe is None:
            timeframe = self.execution_timeframe
            
        if num_bars is None:
            num_bars = self.max_data_points
        
        try:
            # Manejar caso especial de H6
            if timeframe == 'H6':
                return self.get_h6_data(num_bars)
            
            # Para el resto de timeframes, usar MT5 directamente
            mt5_timeframe = self.timeframe_dict.get(timeframe, mt5.TIMEFRAME_H1)
            bars = mt5.copy_rates_from_pos(self.symbol, mt5_timeframe, 0, num_bars)
            
            if bars is None or len(bars) == 0:
                logger.error(f"No se pudieron obtener datos históricos: {mt5.last_error()}")
                raise Exception(f"Error al obtener datos históricos: {mt5.last_error()}")
            
            # Convertir a DataFrame
            df = pd.DataFrame(bars)
            
            # Mostrar las columnas disponibles para propósitos de depuración
            logger.info(f"Columnas disponibles en datos de MT5: {', '.join(df.columns)}")
            
            # Convertir tiempo a formato datetime y establecer como índice
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # En MT5, el volumen está en la columna tick_volume
            if 'tick_volume' in df.columns and 'volume' not in df.columns:
                df['volume'] = df['tick_volume']
                logger.info("Columna 'tick_volume' mapeada a 'volume'")
            
            # Verificar que todas las columnas OHLC estén presentes con sus nombres esperados
            ohlc_mappings = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close'
            }
            
            for orig, renamed in ohlc_mappings.items():
                if orig in df.columns and renamed not in df.columns:
                    df[renamed] = df[orig]
                    logger.info(f"Columna '{orig}' mapeada a '{renamed}'")
            
            # Verificar que todas las columnas requeridas estén presentes
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Si falta 'volume', intentamos crear una columna ficticia (mejor que fallar)
                if 'volume' in missing_columns:
                    logger.warning("Columna 'volume' no encontrada, creando columna ficticia con valores 1")
                    df['volume'] = 1
                    missing_columns.remove('volume')
                
                # Si todavía faltan columnas críticas, lanzamos error
                if missing_columns:
                    raise Exception(f"Columnas requeridas no encontradas: {', '.join(missing_columns)}. Columnas disponibles: {', '.join(df.columns)}")
            
            logger.info(f"Datos históricos obtenidos: {len(df)} barras para timeframe {timeframe}")
            return df
        except Exception as e:
            logger.error(f"Error al obtener datos históricos para {timeframe}: {e}")
            raise

    def prepare_data(self, df):
        """Prepara los datos para el entrenamiento del modelo LSTM"""
        try:
            # Verificar y renombrar columnas si es necesario
            if 'tick_volume' in df.columns and 'volume' not in df.columns:
                df['volume'] = df['tick_volume']
                logger.info("Columna 'tick_volume' renombrada a 'volume'")
            
            # Verificar que todas las columnas necesarias estén presentes
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    available_cols = ', '.join(df.columns)
                    raise Exception(f"La columna '{col}' no está presente en el DataFrame. Columnas disponibles: {available_cols}")
            
            # Seleccionar características y objetivo
            features = df[['open', 'high', 'low', 'close', 'volume']].values
            targets = df[['high', 'low', 'close']].values
            
            # Normalizar datos
            features_scaled = self.scaler_X.fit_transform(features)
            targets_scaled = self.scaler_y.fit_transform(targets)
            
            # Crear secuencias para LSTM
            X, y = [], []
            for i in range(self.look_back, len(features_scaled)):
                X.append(features_scaled[i-self.look_back:i])
                y.append(targets_scaled[i])
            
            X, y = np.array(X), np.array(y)
            
            # Dividir en train y test (80/20)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            logger.info(f"Datos preparados - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error al preparar los datos: {e}")
            raise

    def build_model(self, input_shape):
        """Construye el modelo LSTM con regularización"""
        try:
            from tensorflow.keras.regularizers import l1_l2
            
            model = Sequential()
            
            # Primera capa LSTM con regularización L2
            model.add(LSTM(64, 
                        return_sequences=True, 
                        input_shape=input_shape,
                        kernel_regularizer=l1_l2(l1=0.0, l2=0.001)))
            model.add(Dropout(0.3))  # Aumentar dropout de 0.2 a 0.3
            
            # Segunda capa LSTM con regularización L2
            model.add(LSTM(64, kernel_regularizer=l1_l2(l1=0.0, l2=0.001)))
            model.add(Dropout(0.3))  # Aumentar dropout de 0.2 a 0.3
            
            # Capa de salida (predice high, low, close)
            model.add(Dense(3))
            
            # Compilar modelo
            model.compile(optimizer='adam', loss='mse')
            
            logger.info("Modelo LSTM con regularización construido")
            return model
        except Exception as e:
            logger.error(f"Error al construir el modelo: {e}")
            raise

    def train_model(self):
        """Entrena o reentrea el modelo LSTM con métricas mejoradas"""
        try:
            # Obtener datos históricos del timeframe de predicción (6h)
            df = self.get_historical_data(timeframe=self.prediction_timeframe)
            
            # Preparar datos
            X_train, y_train, X_test, y_test = self.prepare_data(df)
            
            # Construir o cargar modelo
            if os.path.exists(self.model_path) and self.model is None:
                try:
                    self.model = load_model(self.model_path)
                    logger.info("Modelo existente cargado correctamente")
                except:
                    logger.warning("No se pudo cargar el modelo existente, creando uno nuevo")
                    self.model = self.build_model(X_train.shape[1:])
            elif self.model is None:
                self.model = self.build_model(X_train.shape[1:])
            
            # Callbacks para entrenamiento
                        # Callbacks para entrenamiento
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(self.model_path, save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            # Entrenar modelo
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluar modelo
            y_pred_scaled = self.model.predict(X_test)
            y_test_unscaled = self.scaler_y.inverse_transform(y_test)
            y_pred_unscaled = self.scaler_y.inverse_transform(y_pred_scaled)
            
            # Evaluar el modelo con métricas avanzadas
            additional_metrics = self.evaluate_model_performance(y_test_unscaled, y_pred_unscaled)
            
            # Visualizar resultados
            self._plot_training_history(history)
            self._plot_predictions(y_test_unscaled, y_pred_unscaled)
            
            # Actualizar tiempo de entrenamiento
            self.last_trained = datetime.now()
            
            # Calcular métricas de dirección para mensaje
            direction_accuracy = additional_metrics.get('direction_accuracy', 0) if additional_metrics else 0
            
            # Enviar resultados a Telegram
            message = (
                f"🔄 Modelo reentrenado para {self.prediction_timeframe}\n"
                f"MSE: {self.metrics['mse'][-1]:.4f}\n"
                f"RMSE: {self.metrics['rmse'][-1]:.4f}\n"
                f"MAE: {self.metrics['mae'][-1]:.4f}\n"
                f"R²: {self.metrics['r2'][-1]:.4f}\n"
                f"Precisión Dirección: {direction_accuracy:.2f}%\n"
                f"Cantidad de datos: {len(df)}"
            )
            
            # Usar rutas absolutas para los archivos de imagen
            predictions_img = os.path.join(self.script_dir, 'predictions.png')
            self.send_telegram_message(message, predictions_img)
            
            logger.info(f"Modelo entrenado correctamente - MSE: {self.metrics['mse'][-1]:.4f}, MAE: {self.metrics['mae'][-1]:.4f}, R²: {self.metrics['r2'][-1]:.4f}")
            return history
        except Exception as e:
            logger.error(f"Error al entrenar el modelo: {e}")
            raise

    
    def _plot_advanced_metrics(self):
        """Crea visualizaciones avanzadas para las métricas del modelo"""
        try:
            # Verificar si hay suficientes datos
            if len(self.metrics['mse']) < 2:
                logger.info("No hay suficientes datos para graficar métricas avanzadas")
                return
            
            # Crear figura con múltiples subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # 1. Evolución de MSE, RMSE y MAE
            axes[0, 0].plot(self.metrics['mse'], label='MSE', marker='o')
            if 'rmse' in self.metrics and len(self.metrics['rmse']) > 0:
                axes[0, 0].plot(self.metrics['rmse'], label='RMSE', marker='s')
            axes[0, 0].plot(self.metrics['mae'], label='MAE', marker='^')
            axes[0, 0].set_title('Evolución de Errores de Predicción')
            axes[0, 0].set_xlabel('Reentrenamientos')
            axes[0, 0].set_ylabel('Valor')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 2. Evolución de R² y Correlación
            axes[0, 1].plot(self.metrics['r2'], label='R²', marker='o', color='green')
            if 'correlation' in self.metrics and len(self.metrics['correlation']) > 0:
                axes[0, 1].plot(self.metrics['correlation'], label='Correlación', marker='s', color='purple')
            axes[0, 1].set_title('Métricas de Bondad de Ajuste')
            axes[0, 1].set_xlabel('Reentrenamientos')
            axes[0, 1].set_ylabel('Valor')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 3. Precisión de Dirección y MAPE
            ax3 = axes[1, 0]
            
            if 'direction_accuracy' in self.metrics and len(self.metrics['direction_accuracy']) > 0:
                color = 'tab:blue'
                ax3.set_xlabel('Reentrenamientos')
                ax3.set_ylabel('Precisión Dirección (%)', color=color)
                ax3.plot(self.metrics['direction_accuracy'], label='Precisión Dirección', 
                        marker='o', color=color)
                ax3.tick_params(axis='y', labelcolor=color)
                ax3.set_ylim([0, 100])
                
                # Crear eje secundario para MAPE
                ax3b = ax3.twinx()
                color = 'tab:red'
                ax3b.set_ylabel('MAPE (%)', color=color)
                if 'mape' in self.metrics and len(self.metrics['mape']) > 0:
                    ax3b.plot(self.metrics['mape'], label='MAPE', 
                            marker='s', color=color, linestyle='--')
                ax3b.tick_params(axis='y', labelcolor=color)
                
                # Título y leyenda
                ax3.set_title('Precisión de Dirección vs MAPE')
                
                # Combinamos leyendas de ambos ejes
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3b.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 4. Sharpe Ratio
            if 'sharpe_ratio' in self.metrics and len(self.metrics['sharpe_ratio']) > 0:
                axes[1, 1].plot(self.metrics['sharpe_ratio'], label='Sharpe Ratio', 
                            marker='D', color='orange')
                axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.7)
                axes[1, 1].axhline(y=2, color='g', linestyle='--', alpha=0.7)
                axes[1, 1].text(0, 1, 'Aceptable', verticalalignment='bottom')
                axes[1, 1].text(0, 2, 'Bueno', verticalalignment='bottom')
                axes[1, 1].set_title('Sharpe Ratio (Simplificado)')
                axes[1, 1].set_xlabel('Reentrenamientos')
                axes[1, 1].set_ylabel('Ratio')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Guardar con ruta absoluta
            advanced_metrics_img = os.path.join(self.script_dir, 'advanced_metrics.png')
            plt.savefig(advanced_metrics_img)
            plt.close()
            
            # Enviar imagen por Telegram
            self.send_telegram_message("📊 Métricas Avanzadas del Modelo", advanced_metrics_img)
            logger.info(f"Gráfico de métricas avanzadas guardado en {advanced_metrics_img}")
            
        except Exception as e:
            logger.error(f"Error al generar gráficos de métricas avanzadas: {e}")



    


    def _plot_training_history(self, history):
        """Visualiza la historia de entrenamiento"""
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss During Training')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            # Guardar con ruta absoluta
            history_img = os.path.join(self.script_dir, 'training_history.png')
            plt.savefig(history_img)
            plt.close()
        except Exception as e:
            logger.error(f"Error al visualizar la historia de entrenamiento: {e}")

    def _plot_predictions(self, y_true, y_pred, n_samples=100):
        """Visualiza las predicciones vs valores reales"""
        try:
            # Limitar a las últimas n_samples para mejor visualización
            if len(y_true) > n_samples:
                y_true = y_true[-n_samples:]
                y_pred = y_pred[-n_samples:]
            
            # Crear figura
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))
            titles = ['High', 'Low', 'Close']
            
            for i, title in enumerate(titles):
                axes[i].plot(y_true[:, i], label=f'Real {title}', color='blue')
                axes[i].plot(y_pred[:, i], label=f'Predicción {title}', color='red', linestyle='--')
                axes[i].set_title(f'Predicciones vs Reales - {title}')
                axes[i].set_xlabel('Tiempo')
                axes[i].set_ylabel('Precio')
                axes[i].legend()
                axes[i].grid(True)
            
            plt.tight_layout()
            
            # Guardar con ruta absoluta
            predictions_img = os.path.join(self.script_dir, 'predictions.png')
            plt.savefig(predictions_img)
            plt.close()
        except Exception as e:
            logger.error(f"Error al visualizar las predicciones: {e}")

    def _plot_metrics_over_time(self):
        """Visualiza la evolución de las métricas a lo largo del tiempo"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['mse'], label='MSE')
            plt.plot(self.metrics['mae'], label='MAE')
            plt.plot(self.metrics['r2'], label='R²')
            plt.title('Evolución de Métricas')
            plt.xlabel('Reentrenamientos')
            plt.ylabel('Valor')
            plt.legend()
            
            # Guardar con ruta absoluta
            metrics_img = os.path.join(self.script_dir, 'metrics_evolution.png')
            plt.savefig(metrics_img)
            plt.close()
            
            self.send_telegram_message("📊 Evolución de métricas del modelo", metrics_img)
        except Exception as e:
            logger.error(f"Error al visualizar las métricas: {e}")

    def predict_next_candle(self):
        """Predice el próximo valor de high, low, close"""
        try:
            # Verificar si el modelo existe
            if self.model is None:
                logger.warning("No hay modelo para predecir, entrenando uno nuevo")
                self.train_model()
            
            # Obtener los últimos datos
            df = self.get_historical_data(self.look_back + 1)
            
            # Preparar datos para la predicción
            features = df[['open', 'high', 'low', 'close', 'volume']].values
            features_scaled = self.scaler_X.transform(features)
            
            # Crear secuencia para LSTM
            X_pred = np.array([features_scaled])
            
            # Predecir
            y_pred_scaled = self.model.predict(X_pred)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            
            # Obtener valores actuales
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            current_close = df['close'].iloc[-1]
            
            # Calcular porcentajes de cambio
            pred_high, pred_low, pred_close = y_pred[0]
            
            high_change_pct = ((pred_high - current_close) / current_close) * 100
            low_change_pct = ((pred_low - current_close) / current_close) * 100
            close_change_pct = ((pred_close - current_close) / current_close) * 100
            
            logger.info(f"Predicción - High: {pred_high:.5f} ({high_change_pct:.2f}%), "
                        f"Low: {pred_low:.5f} ({low_change_pct:.2f}%), "
                        f"Close: {pred_close:.5f} ({close_change_pct:.2f}%)")
            
            return {
                'pred_high': pred_high,
                'pred_low': pred_low,
                'pred_close': pred_close,
                'high_change_pct': high_change_pct,
                'low_change_pct': low_change_pct,
                'close_change_pct': close_change_pct,
                'current_close': current_close
            }
        except Exception as e:
            logger.error(f"Error al predecir el próximo valor: {e}")
            return None

    def place_order(self, prediction):
        """Coloca una orden basada en la predicción"""
        try:
            # Extraer datos de la predicción
            close_change_pct = prediction['close_change_pct']
            current_close = prediction['current_close']
            
            # Determinar dirección basada en el porcentaje de cambio del precio de cierre
            if abs(close_change_pct) < self.price_change_threshold:
                logger.info(f"No se coloca orden - Cambio de precio ({close_change_pct:.2f}%) por debajo del umbral ({self.price_change_threshold}%)")
                return None
            
            # Determinar tipo de orden
            order_type = mt5.ORDER_TYPE_BUY if close_change_pct > 0 else mt5.ORDER_TYPE_SELL
            direction = "COMPRA" if order_type == mt5.ORDER_TYPE_BUY else "VENTA"
            
            # Calcular stop loss y take profit
            price_info = mt5.symbol_info_tick(self.symbol)
            
            if price_info is None:
                logger.error(f"No se pudo obtener información de precio para {self.symbol}")
                return None
            
            current_price = price_info.ask if order_type == mt5.ORDER_TYPE_BUY else price_info.bid
            
            # Calcular pip value
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"No se pudo obtener información del símbolo {self.symbol}")
                return None
            
            pip_value = 10**(-symbol_info.digits)
            
            # Calcular SL y TP en pips
            atr = self._calculate_atr(20)  # ATR de 20 periodos
            sl_pips = atr * self.sl_multiplier
            tp_pips = atr * self.tp_multiplier
            
            # Convertir pips a precio
            if order_type == mt5.ORDER_TYPE_BUY:
                sl_price = current_price - sl_pips
                tp_price = current_price + tp_pips
            else:
                sl_price = current_price + sl_pips
                tp_price = current_price - tp_pips
            
            # Preparar la solicitud de orden
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": order_type,
                "price": current_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": 12345,
                "comment": f"LSTM Prediction: {close_change_pct:.2f}%",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Enviar la orden
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Error al colocar orden: {result.comment}")
                self.send_telegram_message(f"❌ Error al colocar orden: {result.comment}")
                return None
            
            # Registrar y notificar sobre la orden
            logger.info(f"Orden colocada: {direction} {self.lot_size} {self.symbol} @ {current_price}, SL: {sl_price}, TP: {tp_price}")
            
            # Enviar mensaje a Telegram
            message = (
                f"🔔 NUEVA ORDEN: {direction}\n"
                f"Símbolo: {self.symbol}\n"
                f"Lote: {self.lot_size}\n"
                f"Precio de entrada: {current_price}\n"
                f"Stop Loss: {sl_price}\n"
                f"Take Profit: {tp_price}\n"
                f"Predicción de cambio: {close_change_pct:.2f}%\n"
                f"Predicción High: {prediction['pred_high']:.5f} ({prediction['high_change_pct']:.2f}%)\n"
                f"Predicción Low: {prediction['pred_low']:.5f} ({prediction['low_change_pct']:.2f}%)\n"
                f"Predicción Close: {prediction['pred_close']:.5f} ({prediction['close_change_pct']:.2f}%)"
            )
            self.send_telegram_message(message)
            
            return result
        except Exception as e:
            logger.error(f"Error al colocar la orden: {e}")
            self.send_telegram_message(f"❌ Error al colocar orden: {str(e)}")
            return None

    def _calculate_atr(self, period=14):
        """Calcula el ATR (Average True Range) para el periodo especificado"""
        try:
            # Obtener datos históricos
            df = self.get_historical_data(period + 1)
            
            # Calcular True Range
            df['high-low'] = df['high'] - df['low']
            df['high-prev_close'] = abs(df['high'] - df['close'].shift(1))
            df['low-prev_close'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)
            
            # Calcular ATR
            atr = df['tr'].mean()
            
            return atr
        except Exception as e:
            logger.error(f"Error al calcular ATR: {e}")
            return 0.001  # Valor por defecto pequeño

    def run(self):
        """Ejecuta el bot de trading con órdenes al inicio de cada nueva vela"""
        try:
            logger.info("Iniciando bot de trading...")
            self.send_telegram_message("🚀 Bot de Trading iniciado!")
            
            # Variables para controlar órdenes por vela
            self.last_candle_time = None
            self.order_placed_for_current_candle = False
            
            # Entrenar el modelo inicial
            self.train_model()
            
            while True:
                try:
                    # Verificar si es necesario reentrenar el modelo
                    if (self.last_trained is None or 
                        datetime.now() - self.last_trained > timedelta(hours=self.retraining_hours)):
                        logger.info(f"Reentrenando modelo (último entrenamiento: {self.last_trained})")
                        self.train_model()
                        
                        # Graficar evolución de métricas si hay suficientes datos
                        if len(self.metrics['mse']) > 1:
                            self._plot_metrics_over_time()
                            
                        # Graficar métricas avanzadas si están disponibles
                        if hasattr(self, '_plot_advanced_metrics'):
                            self._plot_advanced_metrics()
                    
                    # Obtener la última vela completada para determinar si estamos en una nueva vela
                    mt5_timeframe = self.timeframe_dict.get(self.execution_timeframe, mt5.TIMEFRAME_H1)
                    last_candle = mt5.copy_rates_from_pos(self.symbol, mt5_timeframe, 0, 1)
                    
                    if last_candle is None or len(last_candle) == 0:
                        logger.error("No se pudo obtener la última vela")
                        time.sleep(60)
                        continue
                    
                    # Convertir tiempo a formato datetime
                    current_candle_time = datetime.fromtimestamp(last_candle[0]['time'])
                    
                    # Verificar si estamos en una nueva vela
                    is_new_candle = (self.last_candle_time is None or 
                                    current_candle_time > self.last_candle_time)
                    
                    # Si estamos en una nueva vela, procesamos inmediatamente
                    if is_new_candle:
                        if self.last_candle_time is not None:
                            logger.info(f"Nueva vela detectada: {current_candle_time} (anterior: {self.last_candle_time})")
                        else:
                            logger.info(f"Primera vela detectada: {current_candle_time}")
                        
                        # Actualizar el tiempo de la última vela
                        self.last_candle_time = current_candle_time
                        
                        # Restablecer el estado de la orden para la nueva vela
                        self.order_placed_for_current_candle = False
                        
                        # Realizar predicción al inicio de la nueva vela
                        prediction = self.predict_next_candle()
                        
                        if prediction:
                            # Enviar mensaje de predicción a Telegram
                            message = (
                                f"🔮 PREDICCIÓN (NUEVA VELA):\n"
                                f"Símbolo: {self.symbol}\n"
                                f"Timeframe: {self.execution_timeframe}\n"
                                f"Vela iniciada: {current_candle_time}\n"
                                f"Predicción High: {prediction['pred_high']:.5f} ({prediction['high_change_pct']:.2f}%)\n"
                                f"Predicción Low: {prediction['pred_low']:.5f} ({prediction['low_change_pct']:.2f}%)\n"
                                f"Predicción Close: {prediction['pred_close']:.5f} ({prediction['close_change_pct']:.2f}%)"
                            )
                            self.send_telegram_message(message)
                            
                            # Determinar si debemos colocar orden basado en el umbral de cambio
                            should_place_order = abs(prediction['close_change_pct']) >= self.price_change_threshold
                            
                            if should_place_order:
                                result = self.place_order(prediction)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    self.order_placed_for_current_candle = True
                                    logger.info(f"Orden colocada al inicio de la vela {current_candle_time}")
                            else:
                                logger.info(f"No se coloca orden - Cambio de precio ({prediction['close_change_pct']:.2f}%) por debajo del umbral ({self.price_change_threshold}%)")
                    else:
                        # Si no es una nueva vela, simplemente registramos que estamos esperando
                        logger.info(f"Esperando nueva vela (actual: {current_candle_time})")
                    
                    # Esperar hasta el próximo periodo usando el método implementado
                    wait_time, _ = self._get_wait_time(self.execution_timeframe)
                    logger.info(f"Esperando {wait_time} segundos hasta la próxima verificación...")
                    time.sleep(wait_time)
                
                except Exception as e:
                    logger.error(f"Error durante la ejecución: {e}")
                    self.send_telegram_message(f"⚠️ Error durante la ejecución: {str(e)}")
                    time.sleep(60)  # Esperar un minuto antes de reintentar
                    
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario")
            self.send_telegram_message("🛑 Bot detenido por el usuario")
        except Exception as e:
            logger.error(f"Error fatal: {e}")
            self.send_telegram_message(f"🚨 ERROR FATAL: {str(e)}")
        finally:
            # Limpiar recursos
            if mt5.initialize():  # Check if MT5 is initialized
                mt5.shutdown()
                logger.info("MT5 desconectado")
            else:
                logger.info("MT5 ya estaba desconectado")

if __name__ == "__main__":
    try:
        # Crear y ejecutar el bot
        bot = MT5LSTMTrader()
        bot.run()
    except Exception as e:
        logging.error(f"Error al iniciar el bot: {e}")