import pandas as pd
import joblib
from pathlib import Path
import sys
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model_car_rental.config import config  
from prediction_model_car_rental.processing.data_handling import load_pipeline

# Load the pre-trained pipeline from the specified model file
price_prediction_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions(data_input):
    """
    Generate predictions using the trained pipeline.

    Parameters:
    data_input (list of dicts): Input data for prediction.

    Returns:
    dict: Dictionary containing the predictions with key "prediction".
    """
    # Convert input data to a DataFrame
    data = pd.DataFrame(data_input)
    
    # Drop unwanted columns
    data = data.drop(columns=config.DROP_FEATURES)
    
    # Use the pipeline to predict the target variable
    pred = price_prediction_pipeline.predict(data)
    
    # Create a dictionary with the prediction results
    result = pred.tolist()
    return result

if __name__ == '__main__':
    # Example usage: generate predictions with sample data
    sample_data = [{'name': 'Maruti Swift Dzire VDI','year': 2014, 'km_driven': 72000, 'fuel': 'Diesel', 'seller_type': 'Individual',
                    'transmission': 'Manual', 'owner': 'First Owner', 'mileage(km/ltr/kg)': 23.59,
                    'engine': 1248, 'max_power':74,'seats': 5}]
    predictions = generate_predictions(sample_data)
    print("The Expected Price of teh Mentioned Model is ",predictions)
