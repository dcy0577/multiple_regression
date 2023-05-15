import pandas as pd
import numpy as np
import os

def split_dataset(ratio, input_path):
    df = pd.read_csv(input_path, header=None, delimiter=r"\s+")
    rng = np.random.RandomState()

    train = df.sample(frac=ratio, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]

    np.savetxt(r"/home/dchangyu/multiple_regression/boston_train.txt", train.values, fmt="%s")
    np.savetxt(r"/home/dchangyu/multiple_regression/boston_test.txt", test.values, fmt="%s")

split_dataset(0.7, "/home/dchangyu/multiple_regression/housing.csv")