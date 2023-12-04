# utils.py

import math
import numpy as np

def formatPrice(n):
    """Format the price for display."""
    return ("-Rs." if n < 0 else "Rs.") + "{0:.2f}".format(abs(n))


def getStockDataVec(key):
    """Get stock data as a vector."""
    vec = []
    lines = open(key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))
    return vec


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-x))


def getState(data, t, n):
    """Get the state for the current time step."""
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])
