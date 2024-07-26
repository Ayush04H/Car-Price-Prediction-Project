from pathlib import Path
import os
import sys

# Add the package root directory to the system path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# Import necessary modules from the prediction_model package
from prediction_model_car_rental.config import config  
from prediction_model_car_rental.processing.data_handling import load_dataset, save_pipeline
import prediction_model_car_rental.pipeline as pipe

def perform_training():
    """
    Load training data, train the price prediction pipeline, and save the trained model.
    """
    # Load the dataset
    data = load_dataset(config.DATA_FILE)
    
    # Separate features and target
    X = data.drop(columns=config.DROP_FEATURES)
    y = data[config.TARGET]
    
    # Train the pipeline
    pipe.price_prediction_pipeline.fit(X, y)
    
    # Save the trained pipeline
    save_pipeline(pipe.price_prediction_pipeline)

if __name__ == '__main__':
    # Perform training when the script is run directly
    perform_training()
