#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the enhanced model evaluation on completely unseen data
"""

import os
import sys
import logging
import json
import traceback
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting model evaluation on unseen data")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Import the ModelEnhancer from the module
        sys.path.append(script_dir)
        from mt5_model_enhancements import ModelEnhancer
        
        # Path to config file
        config_path = os.path.join(script_dir, 'config.json')
        
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at {config_path}")
            return 1
        
        # Initialize the enhancer
        enhancer = ModelEnhancer(config_path)
        
        # Create output directory for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, f"model_evaluation_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the pipeline
        logger.info("Running model enhancement and evaluation pipeline")
        results = enhancer.run_enhancement_pipeline()
        
        # Save detailed results to JSON
        results_file = os.path.join(output_dir, "evaluation_results.json")
        
        # Convert NumPy values to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj
        
        serializable_results = {
            'cross_val_metrics': [{k: convert_to_serializable(v) for k, v in m.items()} 
                                  for m in results['cross_val_metrics']],
            'avg_metrics': {k: convert_to_serializable(v) for k, v in results['avg_metrics'].items()},
            'unseen_metrics': {k: convert_to_serializable(v) for k, v in results['unseen_metrics'].items()}
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        # Copy plots to output directory
        import shutil
        for plot_file in [
            'enhanced_training_history.png',
            'unseen_data_predictions.png',
            'cross_validation_results.png',
            'model_enhancement_report.txt'
        ]:
            src = os.path.join(script_dir, plot_file)
            if os.path.exists(src):
                dst = os.path.join(output_dir, plot_file)
                shutil.copy2(src, dst)
        
        # Print summary of results
        logger.info("\n" + "="*50)
        logger.info("MODEL EVALUATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Cross-validation avg RMSE: {results['avg_metrics']['avg_rmse']:.5f}")
        logger.info(f"Cross-validation avg Direction Accuracy: {results['avg_metrics']['avg_direction_accuracy']:.2f}%")
        logger.info(f"Unseen data RMSE: {results['unseen_metrics']['rmse']:.5f}")
        logger.info(f"Unseen data Direction Accuracy: {results['unseen_metrics']['direction_accuracy']:.2f}%")
        logger.info(f"R² on unseen data: {results['unseen_metrics']['r2']:.5f}")
        
        # Check for overfitting
        cv_direction = results['avg_metrics']['avg_direction_accuracy']
        unseen_direction = results['unseen_metrics']['direction_accuracy']
        direction_drop = cv_direction - unseen_direction
        
        cv_rmse = results['avg_metrics']['avg_rmse']
        unseen_rmse = results['unseen_metrics']['rmse']
        rmse_increase = ((unseen_rmse - cv_rmse) / cv_rmse) * 100
        
        is_overfit = (direction_drop > 10) or (rmse_increase > 20)
        
        if is_overfit:
            logger.warning("\nPOTENTIAL OVERFITTING DETECTED")
            logger.warning(f"Direction accuracy drop: {direction_drop:.2f}%")
            logger.warning(f"RMSE increase: {rmse_increase:.2f}%")
            logger.warning("Model may be overfitting to the training data")
            logger.warning("Consider implementing stronger regularization or reducing model complexity")
        else:
            logger.info("\nNo significant overfitting detected")
            logger.info("Model appears to generalize well to unseen data")
        
        logger.info("\nDetailed results and plots saved to: " + output_dir)
        logger.info("="*50)
        
        # Clean up
        enhancer.cleanup()
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())