import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn

data = pd.read_csv("data/amazon.csv")

close = data[['Close']]
close_scaler = MinMaxScaler(feature_range=(-1, 1))
close['Close'] = close_scaler.fit_transform(close['Close'].values.reshape(-1,1))
close = close.to_numpy().reshape(-1)

high = data[['High']]
high_scaler = MinMaxScaler(feature_range=(-1, 1))
high['High'] = high_scaler.fit_transform(high['High'].values.reshape(-1,1))
high = high.to_numpy().reshape(-1)

LOOKBACK = 20

def split_data():
    total_data_size = len(high) - LOOKBACK
    data = np.zeros((total_data_size, 2, LOOKBACK))
    y = np.zeros(total_data_size)

    for index in range(len(high) - LOOKBACK):
        data[index, 0] = close[index: index + LOOKBACK]
        data[index, 1] = high[index: index + LOOKBACK]
        y[index] = close[index + LOOKBACK]

    test_set_size = int(np.round(0.2 * data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size]
    y_train = y[:train_set_size]

    x_test = data[train_set_size:]
    y_test = y[train_set_size:]

    return [x_train, y_train, x_test, y_test]


def get_data():
    x_train, y_train, x_test, y_test = split_data()
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data()
    print(x_train[0][0])
    print(x_train[0][1])
    print(y_train[0])