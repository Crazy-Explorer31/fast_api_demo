from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import re

from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pickle

def transform_feature_1(s):
    if pd.isna(s):
        return 0
    elif s.split(' ')[0] == '':
        return 0
    return float(s.split(' ')[0])

def transform_feature_2(s):
    if pd.isna(s):
        return (s, s)
    nums_in_s = re.findall(r'\d+\.?\d*', s)
    if len(nums_in_s) < 2:
        return (np.nan, np.nan)
    torque, max_torque_rpm = float(nums_in_s[0]), float(nums_in_s[-1])
    if 'nm' in s.lower():
        torque = torque
    elif 'kgm' in s.lower():
        torque = 10 * torque
    return (torque, max_torque_rpm)
transform_feature_2_torque = lambda s : transform_feature_2(s)[0]
transform_feature_2_max_torque_rpm = lambda s : transform_feature_2(s)[1]

pos_num = {
    'Test' : 0,
    'First' : 1,
    'Second' : 2,
    'Third' : 3,
    'Fourth' : 4
}

class FeaturesGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()

        X_new['mileage'] = X['mileage'].apply(transform_feature_1).astype(float)
        X_new['max_power'] = X['max_power'].apply(transform_feature_1).astype(float)
        X_new['engine'] = X['engine'].apply(transform_feature_1)
        X_new['torque_float'] = X['torque'].apply(transform_feature_2_torque).astype(float)
        X_new['max_torque_rpm'] = X['torque'].apply(transform_feature_2_max_torque_rpm).astype(float)
        X_new['torque'] = X_new['torque_float']
        X_new = X_new.drop(['torque_float'], axis=1)

        # non_target_features = [feature for feature in X_new.columns.to_list() if feature != 'selling_price']
        # X_new = X_new.drop_duplicates(non_target_features, keep='first')

        features_with_nans = ['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']
        X_new[features_with_nans] = X_new[features_with_nans].fillna(X_new[features_with_nans].median())

        X_new['seats'] = X_new['seats'].astype(int).astype(str)
        X_new['engine'] = X_new['engine'].astype(int)

        X_new['name'] = X_new['name'].apply(lambda s : s.split()[0])
        X_new['owner_num'] = X['owner'].apply(lambda s : s.split()[0]).apply(lambda s : pos_num[s])
        X_new['year_squared'] = X['year']**2
        X_new['name_length'] = X['name'].apply(lambda s : len(s))
        X_new['power_per_capacity'] = X_new['max_power'] / X_new['engine']

        return X_new

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = ['torque_float']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

CARS_TRAIN = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
CARS_TEST = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv'

df_train = pd.read_csv(CARS_TRAIN)
df_test = pd.read_csv(CARS_TEST)
df_train = pd.concat([df_train, df_test]) # путь уже до талого учится

numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'owner_num', 'max_torque_rpm', 'year_squared', 'name_length', 'power_per_capacity']
categorical_features = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
target = ['selling_price']

preprocessor_pipline = ColumnTransformer(transformers=[
        ('numerical', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('categorical', OneHotEncoder(drop='first'), categorical_features)
])

regressor_pipline = Pipeline(steps=[
    ('feature_generator', FeaturesGenerator()),
    ('column_dropper', ColumnDropper()),
    ('preprocessor', preprocessor_pipline),
    ('classifier', Ridge(1)) # из части 4
])

regressor_pipline.fit(df_train.drop(['selling_price'], axis=1), df_train[['selling_price']])

model_filename = 'model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(regressor_pipline, model_file)


