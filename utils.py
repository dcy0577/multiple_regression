import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import joblib

def min_max_normalizaion(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data_min_max = scaler.fit_transform(data)
    # save normalization factors
    joblib.dump(scaler, "scaler.pkl")
    return data_min_max


def split_dataset(ratio, input_path):
    df = pd.read_csv(input_path, header=None, delimiter=r"\s+")
    norm_df_value = min_max_normalizaion(df.values[:, 0:12])
    new_df = np.append(norm_df_value, df.values[:, [12,13]], axis=1)
    norm_df = pd.DataFrame(new_df)
    rng = np.random.RandomState()

    train = norm_df.sample(frac=ratio, random_state=rng)
    test = norm_df.loc[~norm_df.index.isin(train.index)]

    np.savetxt(r"/home/dchangyu/multiple_regression/boston_train.txt", train.values, fmt="%s")
    np.savetxt(r"/home/dchangyu/multiple_regression/boston_test.txt", test.values, fmt="%s")

split_dataset(0.7, "/home/dchangyu/multiple_regression/housing.csv")