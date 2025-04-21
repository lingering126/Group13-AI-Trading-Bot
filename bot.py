import numpy as np
import pandas as pd
from datetime import datetime

DATA_PATH = "data/BTC-Daily.csv"

def import_data(path):
    """ 
    Imports Bitcoin price data
    returns: a 2d numpy array with unix timestamp and opening prices as the columns
    """
    df = pd.read_csv(path)
    arr = df[["unix", "open"]].to_numpy()
    return arr

def training_testing_split(data, unix_cutoff):
    training = data[data[:, 0] < unix_cutoff]
    testing = data[data[:, 0] >= unix_cutoff]
    return training, testing



data = import_data(DATA_PATH)
training, testing = training_testing_split(data, datetime(2020, 1, 1).timestamp())
print(training)
print(testing)

