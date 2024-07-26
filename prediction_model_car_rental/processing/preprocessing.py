from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from prediction_model_car_rental.config import config
import pandas as pd

# Preprocessing for numerical data: impute missing values and scale
class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.imputer.fit(X[self.variables])
        self.scaler.fit(X[self.variables])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variables] = self.imputer.transform(X[self.variables])
        X[self.variables] = self.scaler.transform(X[self.variables])
        return X

# Preprocessing for categorical data: impute missing values and one-hot encode
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        self.imputer.fit(X[self.variables])
        self.encoder.fit(X[self.variables])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variables] = self.imputer.transform(X[self.variables])
        return pd.DataFrame(self.encoder.transform(X[self.variables]).toarray(), columns=self.encoder.get_feature_names_out())

# Combine preprocessing steps
def create_preprocessor():
    numeric_features = config.NUM_FEATURES
    categorical_features = config.CAT_FEATURES

    numeric_transformer = NumericalTransformer(variables=numeric_features)
    categorical_transformer = CategoricalTransformer(variables=categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor
