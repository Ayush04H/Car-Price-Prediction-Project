
import os

# Define the root directory of the prediction_model package
PACKAGE_ROOT = "D:\placements2025\Mlops\Car_Rent_Prediction\prediction_model_car_rental"
#print(PACKAGE_ROOT)

# Define the path to the datasets directory within the package
DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")
#print(DATAPATH)

# Define the filename for the dataset
DATA_FILE = 'cardekho.csv'

# Define the name of the model file to be saved
MODEL_NAME = 'car_price_predictor.pkl'

# Define the path to the directory where the trained models will be saved
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')
#print(SAVE_MODEL_PATH)

# Define the target variable for the model
TARGET = 'selling_price'

# List of features to be used in the model
FEATURES = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage(km/ltr/kg)', 'engine', 'seats']

# List of numerical features
NUM_FEATURES = ['year', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'seats']

# List of categorical features
CAT_FEATURES = ['fuel', 'seller_type', 'transmission', 'owner']

# Features to drop
DROP_FEATURES = ['name',  'max_power']
