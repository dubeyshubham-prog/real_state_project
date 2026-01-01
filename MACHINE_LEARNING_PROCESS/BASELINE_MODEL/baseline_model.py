#REQUIRED LIBRARIES=>
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
import warnings

#IMPORTANT CODE SNIPPET TO PRINT THE ENTIRE DATASET:
#SHOWS ALL ROWS
pd.set_option('display.max_rows',None)
#SHOWS ALL COLUMNS
pd.set_option('display.max_columns',None)
#PREVENT LINE WRAPPING
pd.set_option('display.width',None)
#SHOW FULL CONTENT IN EACH CELL
pd.set_option('display.max_colwidth',None)

#IMPORTANT CODE SNIPPET TO IGNORE THE WARNINGS:
warnings.filterwarnings('ignore')

#READING THE DATASET:
df = pd.read_csv('gurgaon_properties_post_feature_selection.csv')
# print(df.head())

#ONE HOT ENCODE-> sector, balcony, agePossession, furnishing type, luxury category, floor category
x = df.drop(columns=['price'])
y = df['price']

#COLUMNS TO ENCODE:
columns_to_encode = ['sector',
                     'balcony',
                     'agePossession',
                     'furnishing_type',
                     'luxury_category',
                     'floor_category']

#APPLYING THE log1p transformation TO THE TARGET VARIABLE:
y_transformed = np.log1p(y)

#CREATING A COLUMN TRANSFORMER FOR PREPROCESSING:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),['property_type', 'bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), columns_to_encode)
    ], remainder='passthrough')
#CREATING A PIPELINE:
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf'))
])

#K-fold CROSS VALIDATION:
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, x, y_transformed, cv=kfold, scoring='r2')
# print(scores.mean())
# print(scores.std())

#TRAINING OF THE DATASET:
x_train, x_test, y_train, y_test = train_test_split(x
                                                    ,y_transformed,
                                                    test_size=0.2,
                                                    random_state=42)
pipeline.fit(x_train, y_train)

#PREDICTION ON THE TEST DATASET:
y_pred = pipeline.predict(x_test)
#TAKING EXPONENTIAL OF THE DATASET:
y_pred = np.expm1(y_pred)
# print(y_pred)

#PERFORMANCE METRIX OF THE MODEL
mse = mean_absolute_error(np.expm1(y_test),y_pred)
# print(mse)