#REQUIRED LIBRARIES=>
import numpy as np
import pandas as pd
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              AdaBoostRegressor)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  Lasso)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   OrdinalEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
import warnings
from sklearn.tree import DecisionTreeRegressor

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
print(df.columns)
# print(df.head())

#COLUMN-->furnishing_type
# print(df['furnishing_type'].value_counts())
# 0->UNFURNISHED
# 1->SEMI FURNISHED
# 2->FURNISHED

df['furnishing_type'] = df['furnishing_type'].replace({0:'unfurnished',
                                                       1:'semifurnished',
                                                       2:'furnished'})
# print(df.head())

#SPLITTING DATASET INTO DEPENDENT AND INDEPENDENT DATASET:
x = df.drop(columns=['price'])
y = df['price']

#APPLYING THE log1p TRANSFORMATION TO THE TARGET VARIABLE:
y_transformed = np.log1p(y)

#PERFORMING ORDINAL ENCODING ON THE REQUIRED COLUMNS:
columns_to_encode = ['property_type',
                     'balcony',
                     'furnishing_type',
                     'luxury_category',
                     'floor_category']

df['sector'] = df['sector'].astype('category')
#------------------------------------- USING ORDINAL ENCODER -------------------------------------------->
'''
#CREATING A COLUMN TRANSFORMER FOR PREPROCESSING:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), columns_to_encode)
    ],
    remainder='passthrough'
)

#CREATING A PIPELINE:
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

#K-fold CROSS VALIDATION:
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, x, y_transformed, cv=kfold, scoring='r2')
print(scores.mean(), scores.std())

#SPLITTING DATASET INTO TRAINING AND TESTING PART:
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y_transformed,
                                                    test_size=0.2,
                                                    random_state=42)

#TRAINING THE DATASET:
pipeline.fit(x_train, y_train)

#PERFORMING PREDICTION ON THE DATASET:
y_pred = pipeline.predict(X_test)

#TAKING EXPONENTIAL OF THE PREDICTED DATASET:
y_pred = np.expm1(y_pred)
mse = mean_absolute_error(np.expm1(y_test),y_pred)
print(mse)

#NOW CREATING A FUNCTION TO AUTOMATE THIS ENTIRE PROCESS:
def scorer(model_name, model):
    output = []
    output.append(model_name)

    # CREATING A PIPELINE:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # K-fold CROSS VALIDATION:
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, x, y_transformed, cv=kfold, scoring='r2')

    output.append(scores.mean())

    # SPLITTING DATASET INTO TRAINING AND TESTING PART:
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y_transformed,
                                                        test_size=0.2,
                                                        random_state=42)

    # TRAINING THE DATASET:
    pipeline.fit(x_train, y_train)

    # PERFORMING PREDICTION ON THE DATASET:
    y_pred = pipeline.predict(X_test)

    output.append(mean_absolute_error(np.expm1(y_test), y_pred))
    return output

#CREATING A DICTIONARY OF ALL REGRESSION MODELS:
model_dict = {
    'linear_reg':LinearRegression(),
    'svr':SVR(),
    'ridge':Ridge(),
    'LASSO':Lasso(),
    'decision tree': DecisionTreeRegressor(),
    'random forest':RandomForestRegressor(),
    'extra trees': ExtraTreesRegressor(),
    'gradient boosting': GradientBoostingRegressor(),
    'adaboost': AdaBoostRegressor(),
    'mlp': MLPRegressor(),
    'xgboost':XGBRegressor()
}

model_output = []
for model_name,model in model_dict.items():
    model_output.append(scorer(model_name, model))

model_output = pd.DataFrame(model_output, columns=['model', 'r2', 'mae'])
print(model_output)
'''
#-------------------------------------------------------------------------------------------------------->

#------------------------------------- USING ONE HOT ENCODER -------------------------------------------->
#CREATING A COLUMN TRANSFORMER FOR PREPROCESSING:
'''
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first', handle_unknown='ignore'),['sector','agePossession','furnishing_type'])
    ]
)

#CREATING A PIPELINE:
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

#K-fold CROSS VALIDATION:
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, x, y_transformed, cv=kfold, scoring='r2')
print('MEAN SCORE = ',scores.mean(),',', 'STANDARD DEVIATION = ',scores.std())

#SPLITTING DATASET INTO  TRAIN AND TEST PART:
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y_transformed,
                                                    test_size=0.2,
                                                    random_state=42)

#TRAINING THE DATASET:
pipeline.fit(x_train,y_train)

#PERFORMING PREDICTION ON THE DATASET:
y_pred = pipeline.predict(x_test)

#TAKING EXPONENTIAL OF THE PREDICTED DATASET:
y_pred = np.expm1(y_pred)
mse = mean_absolute_error(np.expm1(y_test),y_pred)
print(mse)

#CREATING A FUNCTION TO AUTOMATE THIS ENTIRE PROCESS:
def scorer(model_name, model):
    output = []

    output.append(model_name)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, x, y_transformed, cv=kfold, scoring='r2')

    output.append(scores.mean())

    X_train, X_test, y_train, y_test = train_test_split(x, y_transformed, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_pred = np.expm1(y_pred)

    output.append(mean_absolute_error(np.expm1(y_test), y_pred))

    return output

model_dict = {
    'linear_reg':LinearRegression(),
    'svr':SVR(),
    'ridge':Ridge(),
    'LASSO':Lasso(),
    'decision tree': DecisionTreeRegressor(),
    'random forest':RandomForestRegressor(),
    'extra trees': ExtraTreesRegressor(),
    'gradient boosting': GradientBoostingRegressor(),
    'adaboost': AdaBoostRegressor(),
    'mlp': MLPRegressor(),
    'xgboost':XGBRegressor()
}

model_output = []
for model_name,model in model_dict.items():
    model_output.append(scorer(model_name, model))

model_output = pd.DataFrame(model_output, columns=['model', 'r2', 'mae'])
print(model_output)
'''
#-------------------------------------------------------------------------------------------------------->

#------------------------------------- USING ONE HOT ENCODER WITH PCA ----------------------------------->
'''
#CREATING A COLUMN TRANSFORMER FOR PREPROCESSING:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),['sector','agePossession'])
    ],
    remainder='passthrough'
)

#CREATING A PIPELINE
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),
    ('regressor', LinearRegression())
])

#K-fold CROSS VALIDATION:
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline,
                         x,
                         y_transformed,
                         cv=kfold,
                         scoring='r2')
# print('MEAN SCORE = ',scores.mean(),',', 'STANDARD DEVIATION = ',scores.std())

#CREATING A FUNCTION TO AUTOMATE THIS ENTIRE PROCESS:
def scorer(model_name, model):
    output = []

    output.append(model_name)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=0.95)),
        ('regressor', model)
    ])

    # K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, x, y_transformed, cv=kfold, scoring='r2')

    output.append(scores.mean())

    X_train, X_test, y_train, y_test = train_test_split(x, y_transformed, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_pred = np.expm1(y_pred)

    output.append(mean_absolute_error(np.expm1(y_test), y_pred))

    return output

model_dict = {
    'linear_reg':LinearRegression(),
    'svr':SVR(),
    'ridge':Ridge(),
    'LASSO':Lasso(),
    'decision tree': DecisionTreeRegressor(),
    'random forest':RandomForestRegressor(),
    'extra trees': ExtraTreesRegressor(),
    'gradient boosting': GradientBoostingRegressor(),
    'adaboost': AdaBoostRegressor(),
    'mlp': MLPRegressor(),
    'xgboost':XGBRegressor()
}

model_output = []
for model_name,model in model_dict.items():
    model_output.append(scorer(model_name, model))

model_output = pd.DataFrame(model_output, columns=['model', 'r2', 'mae'])
print(model_output)
'''
#-------------------------------------------------------------------------------------------------------->

#------------------------------------------- USING TARGET ENCODER --------------------------------------->
'''
#CREATING A COLUMN TRANSFORMER FOR PREPROCESSING:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),['agePossession']),
        ('target_enc', ce.TargetEncoder(verbose=0), ['sector'])
    ],
    remainder='drop'
)

#CREATING A PIPELINE:
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# K-fold CROSS VALIDATION:
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline,
                         x,
                         y_transformed,
                         cv=kfold,
                         scoring='r2')
print('MEAN SCORE = ',scores.mean(),',', 'STANDARD DEVIATION = ',scores.std())

#CREATING A PROCESS TO AUTOMATE THIS ENTIRE PROCESS:
def scorer(model_name, model):
    output = []

    output.append(model_name)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, x, y_transformed, cv=kfold, scoring='r2')

    output.append(scores.mean())

    X_train, X_test, y_train, y_test = train_test_split(x, y_transformed, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_pred = np.expm1(y_pred)

    output.append(mean_absolute_error(np.expm1(y_test), y_pred))

    return output

model_dict = {
    'linear_reg':LinearRegression(),
    'svr':SVR(),
    'ridge':Ridge(),
    'LASSO':Lasso(),
    'decision tree': DecisionTreeRegressor(),
    'random forest':RandomForestRegressor(),
    'extra trees': ExtraTreesRegressor(),
    'gradient boosting': GradientBoostingRegressor(),
    'adaboost': AdaBoostRegressor(),
    'mlp': MLPRegressor(),
    'xgboost':XGBRegressor()
}

model_output = []
for model_name,model in model_dict.items():
    model_output.append(scorer(model_name, model))

model_df = pd.DataFrame(model_output, columns=['name','r2','mae'])
print(model_df.sort_values(['mae']))
'''
#-------------------------------------------------------------------------------------------------------->

#---------------------------------- USING HYPER-PARAMETER TUNING METHOD --------------------------------->
'''
#DEFINING PARAMETERS FOR THE TRAINING OF THE DATASET:
param_grid = {
    'regressor__n_estimators': [50, 100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__max_samples':[0.1, 0.25, 0.5, 1.0],
    'regressor__max_features': ['auto', 'sqrt']
}

#CREATING A COLUMN TRANSFORMER FOR PRE-PROCESSING:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first',sparse_output=False),['agePossession']),
        ('target_enc', ce.TargetEncoder(), ['sector'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='r2', n_jobs=-1, verbose=4)
search.fit(x, y_transformed)

final_pipe = search.best_estimator_
print(search.best_params_)

#TRAINING THE MODEL:
# final_pipe.fit(x,y_transformed)
'''
#-------------------------------------------------------------------------------------------------------->

#-------------------------- EXPORTING THE BEST POSSIBLE MODEL FOR THE PROJECT --------------------------->
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first',sparse_output=False),['sector','agePossession'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50))
])

pipeline.fit(x,y_transformed)

import pickle
with open('../../pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

with open('df.pkl', 'wb') as file:
    pickle.dump(x, file)

print(x.head())
#-------------------------------------------------------------------------------------------------------->

#TRYING OUT PREDICTIONS:
print(x.columns)
print(x.iloc[0].values)

data = [['house', 'sector 102', 4, 3, '3+', 'New Property', 2750, 0, 0, 'unfurnished', 'Low', 'Low Floor']]
columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
       'agePossession', 'built_up_area', 'servant room', 'store room',
       'furnishing_type', 'luxury_category', 'floor_category']

#CONVERT TO DATAFRAME:
one_df = pd.DataFrame(data, columns=columns)
print(one_df)

#PREDICTIONG ON NEW DATASET:
print(np.expm1(pipeline.predict(one_df)))
print(x.dtypes)










