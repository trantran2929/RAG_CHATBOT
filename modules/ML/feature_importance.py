import pandas as pd

def sarimax_feature_importance(fit):

    params = fit.params

    data = []

    for name, value in params.items():

        # bỏ các tham số AR, variance
        if "sigma2" in name or "ar." in name or "ma." in name:
            continue

        data.append({
            "feature": name,
            "importance": abs(value)
        })

    df = pd.DataFrame(data)

    df = df.sort_values("importance", ascending=False)

    return df

import matplotlib.pyplot as plt


def plot_feature_importance(df):

    plt.figure(figsize=(10,6))

    plt.barh(df["feature"], df["importance"])

    plt.gca().invert_yaxis()

    plt.title("SARIMAX Feature Importance")

    plt.show()