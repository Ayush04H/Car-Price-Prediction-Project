from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from prediction_model_car_rental.processing.preprocessing import create_preprocessor

# Create the preprocessing pipeline
preprocessor = create_preprocessor()

# Define the model
model = RandomForestRegressor(random_state=0)

# Create and evaluate the pipeline
price_prediction_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('model', model)])
