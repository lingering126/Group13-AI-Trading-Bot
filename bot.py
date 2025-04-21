import numpy as np
import pandas as pd
from datetime import datetime

DATA_PATH = "data/BTC-Daily.csv"

def import_data(path):
    """ 
    Imports Bitcoin price data
    returns: a 2d numpy array with unix timestamp and opening prices as the rows
    """
    df = pd.read_csv(path)
    arr = df[["unix", "close"]].to_numpy().T # Transpose so timestamp and price are rows
    arr = arr[:, arr[0,:].argsort()]        # Sort in ascending order of the first row (timestamp)
    return arr

def training_testing_split(data, unix_cutoff):
    training = data[:, data[0] < unix_cutoff]
    testing = data[:, data[0] >= unix_cutoff]
    return training, testing

def pad(price, days):
    """ Pads the price array as done in the task sheet """
    padding = -np.flip(price[1:days])
    return np.append(padding, price)

def WMA(price, days, kernel):
    price = pad(price, days)
    wma = np.correlate(price, kernel, mode="valid")
    print(kernel, sum(kernel))
    return wma

def SMA(price, days):
    """Simple moving average"""
    K = 1/days * np.ones(days)
    return WMA(price, days, K)

def LMA(price, days):
    """Linear moving average"""
    K = 2 * np.linspace(0, 1/days, days)
    return WMA(price, days, K)

def EMA(price, days, alpha):
    """Exponential moving average"""
    K = alpha * np.power(1 - alpha, np.linspace(0, days, days)[::-1])
    K/= sum(K)
    return WMA(price, days, K)

data = import_data(DATA_PATH)
training_data, testing_data = training_testing_split(data, datetime(2020, 1, 1).timestamp())

wma = EMA(training_data[1], 100, 2/101)

from matplotlib import pyplot as plt
plt.plot(training_data[0], training_data[1], label="y")
plt.plot(training_data[0], wma, label="WMA", linestyle="--")
plt.legend()
plt.title("Training Data and EMA")
plt.xlabel("Index")
plt.ylabel("Price")
plt.show()

