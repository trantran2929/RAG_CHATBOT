import numpy as np

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    """
    y_true = np.array(y_true, dtype="float64")
    y_pred = np.array(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    """
    y_true = np.array(y_true, dtype="float64")
    y_pred = np.array(y_pred, dtype="float64")
    return float(np.mean(np.abs(y_true - y_pred)))
