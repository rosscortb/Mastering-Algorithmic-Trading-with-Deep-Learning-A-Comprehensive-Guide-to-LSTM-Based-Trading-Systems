import logging
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
