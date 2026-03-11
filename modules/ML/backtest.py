import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.api.stock_api import get_close_series
from modules.ML.pipeline import train_gap_model


def backtest_gap_model(symbol, test_days=60):

    # lấy dữ liệu giá
    close = get_close_series(symbol, days=365 + test_days)

    # returns
    r = np.log(close / close.shift(1)).dropna()

    preds = []
    actuals = []
    dates = []

    for i in range(len(r) - test_days, len(r)):

        train_r = r.iloc[:i]

        fit, meta, report = train_gap_model(symbol)

        pred = report["ret_hat_next"]

        actual = r.iloc[i]

        preds.append(pred)
        actuals.append(actual)
        dates.append(r.index[i])

    df = pd.DataFrame({
        "date": dates,
        "pred": preds,
        "actual": actuals
    }).set_index("date")

    # RMSE
    rmse = np.sqrt(np.mean((df["pred"] - df["actual"]) ** 2))

    # MAE
    mae = np.mean(np.abs(df["pred"] - df["actual"]))

    # Directional accuracy
    dir_pred = np.sign(df["pred"])
    dir_actual = np.sign(df["actual"])

    direction_acc = (dir_pred == dir_actual).mean()

    print("\n===== BACKTEST RESULT =====")

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Directional accuracy:", direction_acc)

    return df

def plot_prediction(df):

    plt.figure(figsize=(12,5))

    plt.plot(df.index, df["actual"], label="Actual return")
    plt.plot(df.index, df["pred"], label="Predicted return")

    plt.legend()
    plt.title("Predicted vs Actual Returns")

    plt.show()