import logging
import os
import sys

def setup_logging(log_file='trading_bot.log', console_level=logging.INFO, file_level=logging.DEBUG):
    """Setup proper logging with both console and file handlers"""
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set root logger to debug level
    
    # Remove any existing handlers (to avoid duplicates)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log initial message
    logger.info(f"Logging configured - Console level: {logging.getLevelName(console_level)}, File level: {logging.getLevelName(file_level)}")
    logger.info(f"Log file: {os.path.abspath(log_file)}")
    
    return logger

def fix_logging_in_mt5_trader():
    """Patch to fix logging in the MT5LSTMTrader class"""
    import inspect
    
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    # Path to the mt5_lstm_trader.py file
    trader_file = os.path.join(current_dir, 'mt5_lstm_trader.py')
    
    if not os.path.exists(trader_file):
        print(f"Could not find mt5_lstm_trader.py in {current_dir}")
        return False
    
    # Read the file with multiple encoding attempts
    content = None
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            with open(trader_file, 'r', encoding=encoding) as file:
                content = file.read()
            print(f"Successfully read file using {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding, trying another...")
    
    if content is None:
        print("Could not read the file with any encoding")
        return False
    
    # Check if we need to modify the file
    if "warnings.filterwarnings('ignore')" in content and "logging.basicConfig" in content:
        # Replace the problematic logging setup
        updated_content = content.replace(
            "warnings.filterwarnings('ignore')\nlogging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')\nlogger = logging.getLogger(__name__)",
            """warnings.filterwarnings('ignore')

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

logger = _setup_logger()"""
        )
        
        # Write the updated content back to the file
        try:
            # Use the same encoding that successfully read the file
            with open(trader_file, 'w', encoding=encoding) as file:
                file.write(updated_content)
            
            print(f"Successfully updated logging configuration in {trader_file}")
            return True
        except Exception as e:
            print(f"Error writing back to file: {e}")
            return False
    else:
        print("The file has already been modified or has a different structure")
        return False

def patch_run_bot():
    """Patch the run-bot.py file to use proper logging"""
    import inspect
    
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    # Path to the run-bot.py file
    run_bot_file = os.path.join(current_dir, 'run-bot.py')
    
    if not os.path.exists(run_bot_file):
        print(f"Could not find run-bot.py in {current_dir}")
        return False
    
    # Read the file with multiple encoding attempts
    content = None
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            with open(run_bot_file, 'r', encoding=encoding) as file:
                content = file.read()
            print(f"Successfully read run-bot.py using {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"Failed to read run-bot.py with {encoding} encoding, trying another...")
    
    if content is None:
        print("Could not read run-bot.py with any encoding")
        return False
    
    # Check if we need to modify the file
    if "logging.basicConfig(" in content:
        # Replace the problematic logging setup
        updated_content = content.replace(
            "    # Configurar registro\n    log_path = os.path.join(current_dir, \"trading_bot.log\")\n    logging.basicConfig(\n        level=logging.INFO,\n        format='%(asctime)s - %(levelname)s - %(message)s',\n        handlers=[\n            logging.FileHandler(log_path),\n            logging.StreamHandler(sys.stdout)\n        ]\n    )",
            """    # Configurar registro mejorado
    log_path = os.path.join(current_dir, "trading_bot.log")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)"""
        )
        
        # Write the updated content back to the file
        try:
            # Use the same encoding that successfully read the file
            with open(run_bot_file, 'w', encoding=encoding) as file:
                file.write(updated_content)
            
            print(f"Successfully updated logging configuration in {run_bot_file}")
            return True
        except Exception as e:
            print(f"Error writing back to file: {e}")
            return False
    else:
        print("The run-bot.py file has already been modified or has a different structure")
        return False

def create_logging_module():
    """Create a separate logging module file that can be imported"""
    import inspect
    
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    # Path to the new logging module file
    logging_module_file = os.path.join(current_dir, 'bot_logging.py')
    
    # Content for the new module
    content = """import logging
import os
import sys

def setup_logger(name=None, log_file='trading_bot.log', 
                 console_level=logging.INFO, 
                 file_level=logging.DEBUG):
    '''Setup a logger with both console and file handlers'''
    
    # Get logger (root or named)
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger()  # Root logger
    
    logger.setLevel(logging.DEBUG)  # Set to lowest level
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Example usage:
# from bot_logging import setup_logger
# logger = setup_logger(__name__)
# logger.info("This is an info message")
# logger.debug("This is a debug message")
"""
    
    # Write the file
    try:
        with open(logging_module_file, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"Created logging module at {logging_module_file}")
        return True
    except Exception as e:
        print(f"Error creating logging module: {e}")
        return False

def create_simple_fix_script():
    """Create a simple script to add logging directly to trading_bot.log without patching"""
    import inspect
    
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    # Path to the new script
    script_file = os.path.join(current_dir, 'enable_logging.py')
    
    # Content for the new script
    content = """import logging
import os
import sys

# Configure comprehensive logging
log_file = 'trading_bot.log'

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create formatters
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

# Create file handler
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Logging enabled and configured correctly")
logger.info(f"Log file: {os.path.abspath(log_file)}")
logger.info("Run your trading bot normally, logging will now work correctly")
"""
    
    # Write the file
    try:
        with open(script_file, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"Created simple logging fix script at {script_file}")
        return True
    except Exception as e:
        print(f"Error creating simple fix script: {e}")
        return False

if __name__ == "__main__":
    # Setup logging for this script
    log = setup_logging('logging_fix.log')
    
    try:
        log.info("Starting logging system fix")
        
        # Create stand-alone logging module
        module_created = create_logging_module()
        if module_created:
            log.info("Created stand-alone logging module (bot_logging.py)")
        
        # Create simple fix script
        simple_script_created = create_simple_fix_script()
        if simple_script_created:
            log.info("Created simple logging fix script (enable_logging.py)")
        
        # Try to fix logging in MT5LSTMTrader
        try:
            trader_fixed = fix_logging_in_mt5_trader()
            if trader_fixed:
                log.info("Successfully patched MT5LSTMTrader logging")
            else:
                log.warning("Could not patch MT5LSTMTrader logging")
        except Exception as e:
            log.error(f"Error while trying to fix MT5LSTMTrader: {e}", exc_info=True)
            log.info("Using the stand-alone logging module is recommended instead")
        
        # Try to fix logging in run-bot.py
        try:
            run_bot_fixed = patch_run_bot()
            if run_bot_fixed:
                log.info("Successfully patched run-bot.py logging")
            else:
                log.warning("Could not patch run-bot.py logging")
        except Exception as e:
            log.error(f"Error while trying to fix run-bot.py: {e}", exc_info=True)
            log.info("Using the stand-alone logging module is recommended instead")
        
        log.info("""
INSTRUCTIONS:
1. Run 'enable_logging.py' before starting your trading bot
2. OR import and use the logging module in your code:
   from bot_logging import setup_logger
   logger = setup_logger(__name__)
3. Check that 'trading_bot.log' is being created and populated
""")
        
    except Exception as e:
        log.error(f"Error fixing logging system: {e}", exc_info=True)