import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/car_data.csv')


categorical_cols = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
numeric_cols     = ['Year', 'Engine Size', 'Mileage', 'Price']

def simple_imputation(df, numeric_cols, categorical_cols):
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])


def KNN_imputation(df, numeric_cols, categorical_cols):
    from sklearn.impute import KNNImputer
    
    encoders = {}
    for col in categorical_cols:
        codes, unique_vals = pd.factorize(df[col], sort=True)
        encoders[col] = unique_vals
        df[col] = pd.Series(codes, index=df.index)
    
    imputer = KNNImputer(n_neighbors=5)
    cols_to_impute = numeric_cols + categorical_cols
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
    
    for col in categorical_cols:
        df[col] = np.clip(df[col], 0, len(encoders[col]) - 1)
        df[col] = df[col].round().astype(int)
        df[col] = encoders[col][df[col]]


def iterative_imputation(df, numeric_cols, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        codes, unique_vals = pd.factorize(df[col], sort=True)
        encoders[col] = unique_vals
        df[col] = pd.Series(codes, index=df.index)

    imputer = IterativeImputer(
        max_iter=10,      
        random_state=42,
        verbose=0
    )

    cols_to_impute = numeric_cols + categorical_cols
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    for col in categorical_cols:
        df[col] = np.clip(df[col], 0, len(encoders[col]) - 1)
        df[col] = df[col].round().astype(int)
        df[col] = encoders[col][df[col]]