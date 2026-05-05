import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer   
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

df = pd.read_csv('data/car_data.csv')

numeric_cols = ["Price", "Mileage", "Engine Size", "Year"]

def IQR_method(df, col):
    data = df[col].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)].index

def z_score_method(df, col):
    data = df[col].dropna()
    z_scores = zscore(data)
    outlier_indices = data[(z_scores < -3) | (z_scores > 3)].index
    return outlier_indices

def winsorization(df, col):
    data = df[col].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

def clipping(df, col):
    data = df[col].dropna()
    lower_bound = data.quantile(0.01)
    upper_bound = data.quantile(0.99)
    df[col] = np.clip(df[col], lower_bound, upper_bound)