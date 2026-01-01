#REQUIRED LIBRARIES=>
from trace import PRAGMA_NOCOVER

import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import (Lasso,
                                  LinearRegression)
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import re

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
df = pd.read_csv('gurgaon_properties_missing_value_imputation.csv')
print(df.shape)

#DROPPING TWO UNNECESSARY COLUMNS-->society,price_per_sqft
train_df = df.drop(columns=['society','price_per_sqft'])
print(train_df.shape)

#COLUMN-->luxury score
# sns.boxplot(x=df['luxury_score'])

#DEFINING A FUNCTION WHERE WE WILL CATEGORISE THE luxury_score INTO DIFFERENT CATEGORY:
def categorize_luxury(score):
    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None# OR 'UNDEFINED' OR ANY OTHER LABEL FOR SCORES OUTSIDE THE DEFINED BINS
train_df['luxury_category'] = train_df['luxury_score'].apply(categorize_luxury)
# print(train_df['luxury_category'])

#COLUMN-->floorNum
# sns.boxplot(x=df['floorNum'])

#DEFINING A FUNCTION WHERE WE WILL CATEGORISE THE floorNum INTO DIFFERENT CATEGORY:
def categorize_floor(floor):
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None # OR 'UNDEFINED' OR ANY OTHER LABEL FOR SCORES OUTSIDE THE DEFINED BINS

train_df['floor_category'] = train_df['floorNum'].apply(categorize_floor)

#DROPPING BOTH THE COLUMNS luxury_score AND floorNum
train_df.drop(columns=['floorNum','luxury_score'],inplace=True)
# print(train_df.head())

#NOW WE ARE GOING TO ENCODE OUR COLUMNS:
#CREATE A COPY OF THE ORIGINAL DATA FOR LABEL ENCODING:
data_label_encoded = train_df.copy()
#SELECTING THE CATEGORICAL COLUMNS:
categorical_cols = train_df.select_dtypes(include=['object']).columns
# print(categorical_cols)

#APPLY LABEL ENCODING TO CATEGORICAL COLUMNS:
for col in categorical_cols:
    oe = OrdinalEncoder()
    data_label_encoded[col] = oe.fit_transform(data_label_encoded[[col]])
    # print(oe.categories_)

#SPLITTING THE DATASET INTO TRAINING AND TESTING SETS:
x_label = data_label_encoded.drop('price', axis=1)
y_label = data_label_encoded['price']
# print(x_label.head())
# print(y_label.head())

#TECHNIQUE_1->CORRELATION ANALYSIS:
# sns.heatmap(data_label_encoded.corr())

fi_df1 = data_label_encoded.corr()['price'].to_frame().reset_index().rename(columns={'index':'feature','price':'corr_coeff'})
# print(fi_df1)

#TECHNIQUE_2->RANDOM FOREST FEATURE IMPORTANCE:
#TRAIN A RANDOM FOREST REGRESSOR ON LABEL ENCODED DATA:
rf_label = RandomForestRegressor(n_estimators=100, random_state=42)
rf_label.fit(x_label, y_label)
#EXTRACT FEATURE IMPORTANCE SCORE FOR LABEL ENCODED DATA:
fi_df2 = pd.DataFrame({
    'feature': x_label.columns,
    'rf_importance': rf_label.feature_importances_
}).sort_values(by='rf_importance', ascending=False)
# print(fi_df2)

#TECHNIQUE_3->GRADIENT BOOSTING FEATURE IMPORTANCE:
#TRAINA A RANDOM FOREST REGRESSOR ON LABEL ENCODED DATA:
gb_label = GradientBoostingRegressor()
gb_label.fit(x_label, y_label)
#EXTRACT FEATURE IMPORTANCE SCORE FOR LABEL ENCODED DATA:
fi_df3 = pd.DataFrame({
    'feature': x_label.columns,
    'gb_importance': gb_label.feature_importances_
}).sort_values(by='gb_importance', ascending=False)
# print(fi_df3)

#TECHNIQUE_4->PERMUTATION IMPORTANCE:
X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(x_label,
                                                                            y_label,
                                                                            test_size=0.2,
                                                                            random_state=42)
#TRAIN A RANDOM FOREST REGRESSOR ON LABEL ENCODED DATA:
rf_label = RandomForestRegressor(n_estimators=100, random_state=42)
rf_label.fit(X_train_label,y_train_label)

#CALCULATE PERMUTATION IMPORTANCE:
perm_importance = permutation_importance(rf_label, X_test_label, y_test_label, n_repeats=30, random_state=42)

#ORGANIZE RESULTS INTO A DATAFRAME:
fi_df4 = pd.DataFrame({
    'feature': x_label.columns,
    'permutation_importance': perm_importance.importances_mean
}).sort_values(by='permutation_importance', ascending=False)
# print(fi_df4)

#TECHNIQUE_5-->LASSO
#STANDARDIZE THE FEATURES:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x_label)

#TRAIN A LASSO REGRESSIVE MODEL:
#WE'LL USE A RELATIVELY SMALL VALUE FOR ALPHA (THE REGULARIZATION STRENGTH) FOR DEMONSTRATION PURPOSE:
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_scaled, y_label)

#EXTRACT COEFFICIENT:
fi_df5 = pd.DataFrame({
    'feature': x_label.columns,
    'lasso_coeff': lasso.coef_
}).sort_values(by='lasso_coeff', ascending=False)

#TECHNIQUE_6-->RFE
#INITIALIZE THE BASE ESTIMATOR:
estimator = RandomForestRegressor()

#APPLY RFE ON THE LABEL-ENCODED AND STANDARDIZE TRAINING DATA:
selector_label = RFE(estimator, n_features_to_select=x_label.shape[1], step=1)
selector_label = selector_label.fit(x_label, y_label)

#GET THE SELECTED FEATURES BASED ON RFE:
selected_features = x_label.columns[selector_label.support_]

#EXTRACT THE COEFFICIENTS FOR THE SELECTED FEATURES FROM THE UNDERLYING LINEAR REGRESSION MODEL:
selected_coefficients = selector_label.estimator_.feature_importances_

#ORGANIZE THE RESULTS INTO A DATAFRAME:
fi_df6 = pd.DataFrame({
    'feature': selected_features,
    'rfe_score': selected_coefficients
}).sort_values(by='rfe_score', ascending=False)
# print(fi_df6)

#TECHNIQUE_7-->LINEAR REGRESSION WEIGHTS:
# Train a linear regression model on the label-encoded and standardized training data
lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y_label)

# Extract coefficients
fi_df7 = pd.DataFrame({
    'feature': x_label.columns,
    'reg_coeffs': lin_reg.coef_
}).sort_values(by='reg_coeffs', ascending=False)
# print(fi_df7)

#MERGE ALL THE TECHNIQUES TOGETHER:
final_fi_df = fi_df1.merge(fi_df2,on='feature').merge(fi_df3,on='feature').merge(fi_df4,on='feature').merge(fi_df5,on='feature').merge(fi_df6,on='feature').merge(fi_df7,on='feature').set_index('feature')
# print(final_fi_df)

#NORMALIZE THE SCORE:
final_fi_df = final_fi_df.divide(final_fi_df.sum(axis=0), axis=1)
# print(final_fi_df[['rf_importance','gb_importance','permutation_importance','rfe_score']].mean(axis=1).sort_values(ascending=False))

#TO DROP POOJA ROOM, STUDY ROOM, OTHERS:
# print(x_label.head())

#WITH ALL THE COLS:
rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores1 = cross_val_score(rf, x_label, y_label, cv=5, scoring='r2')
# print(scores1.mean())

#AFTER DROPPING THESE 3 COLUMNS POOJA ROOM, STUDY ROOM, OTHERS:
scores2 = cross_val_score(rf, x_label.drop(columns=['pooja room', 'study room', 'others']),
                         y_label,
                         cv=5,
                         scoring='r2')
print(scores2.mean())
export_df = x_label.drop(columns=['pooja room', 'study room', 'others'])
export_df['price'] = y_label
export_df.to_csv('gurgaon_properties_post_feature_selection.csv', index=False)
print(export_df.shape)


























plt.show()