'''
Data pipeline in Python for feature engineering
automate the process repeatable and reliable for forecasting and regression

pipeline with time series data
create lagged features to allow to capture trends and dependencies over time 
'''

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor

''' custom transformer for lagged features ''' 
class LagFeatures(BaseEstimator, TransformerMixin):
  def __init__(self, lag_features=3):
    super().__init__()
    self.lag_features = lag_features

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    ##n_samples, n_features = X.shape
    df = pd.DataFrame(X.copy())
    for lag in range(1, self.lag_features + 1):
      ##df[f'lag_{lag}'] = df.iloc[:, 0].shift(lag)
      df[f'lag_{lag}'] = df['Value'].shift(lag)

    return df

date_range = pd.date_range(start='1/1/2020', periods=100, freq='D')
data = pd.DataFrame(
  {'Date': date_range, 'Value': np.random.randn(len(date_range)).cumsum()})
data.set_index('Date', inplace=True)

lagged_features_data = LagFeatures(lag_features=3).transform(data)
X = lagged_features_data.drop('Value', axis=1)
y = lagged_features_data['Value']

X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, shuffle=False, random_state=42)

''' define the pipeline with scaling, feature scaling and model'''
pipeline = Pipeline([
##  ('scaler', StandardScaler()), ('model', LinearRegression()) ])
  ('scaler', StandardScaler()),
  ('model', XGBRegressor(objective='reg:squarederror')) ])

''' fit the pipeline on the training data'''
pipeline.fit(X_train, y_train)
''' predictions '''
y_pred = pipeline.predict(X_test)

''' evaluate the time series model '''
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.2f}')

''' pipeline imputer, custom transformer '''
class Transformer_Imputer(BaseEstimator, TransformerMixin):
  def __init__(self, missing_data, freq_category):
    self.missing_data = missing_data
    self.freq_category = freq_category
    self.freq_categories = {}

  def fit(self, X, y=None):
    # most frequent category with few missing data
    for var in self.missing_category:
      # self.freq_categories[var] = X[var].value_counts().index[0]
      self.freq_categories[var] = X[var].mode()[0]
    return self

  def transform(self, X,y=None):
    X[self.missing_data] = X[self.missing_data].fillna('Missing')
    for var in self.freq_category:
      X[var].fillna(self.freq_categories[var], inplace=True)
    return X

''' prepare the dataset with the categorical features '''
numeric_features = ['col1', 'col2']
categorical_features = ['col3', 'col4', 'col5']
category_vars_with_na = [
    var for var in categorical_features if X[var].isnull().sum() > 0]
missing_data = [
    var for var in category_vars_with_na if X[var].isnull().mean() > 0.1]
freq_category = [
    var for var in category_vars_with_na if X[var].isnull().mean() < 0.1]
target = 'col6'

X = X.drop(target, axis=1)
y = X[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

''' build the pipeline '''
numeric_transformer = Pipeline(stteps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
 ])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
 ])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_transformer),
        ('cat', categorical_transformer, categorical_transformer)
    ])

pipeline = Pipeline(steps=[
    ('missing_imputer', Transformer_Imputer(missing_data, freq_category))
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(objective='regsquarederror'))])

''' Evaluate the model '''
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

result = pd.DataFrame({'actual': y_test, 'predicted': y_pred}, index=X_test.index)
print(result.head())