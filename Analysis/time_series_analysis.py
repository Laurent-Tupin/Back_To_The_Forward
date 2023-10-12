import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Data_processing.get_market_data import load_stock_price


def SMA_window_with_lowest_MSE(ticker, stock_price_col, length_train, min_window_length, n, API=True):
    """ Find window length with lowest MSE by looping over n different lengths of windows

    Parameters:
        ticker: ticker of the stock
        stock_price_col: open, high, low or close
        length_train: length of the training set
        min_window_length: minimum length of window
        n: size of loop
        API: use API data by default, otherwise loads from local directory


    Returns:
        int: length of window
        float: MSE
    """

    stock = load_stock_price(length_train, ticker, API)[stock_price_col]
    train = stock.iloc[0:(length_train - 1)]
    MSE_list = []
    for i in range(min_window_length, n):
        window = range((length_train - i), (length_train - 1))
        test = stock.iloc[window]
        sma_pred = train.rolling(window=i).mean()
        MSE = np.sqrt(mean_squared_error(sma_pred.iloc[window], test))
        MSE_list = MSE_list + [MSE]

    # TODO CY or LT: how can we write this in a neater way?
    MSE_df = pd.DataFrame({'window_length': range(min_window_length, n),
                           'MSE': MSE_list})
    min_MSE = MSE_df['MSE'] == np.min(MSE_df['MSE'])
    return int(MSE_df.loc[min_MSE, 'window_length'].iloc[0]), np.min(MSE_df['MSE'])


def plot_two_SMA(stock: pd.Series, long_mean: int, short_mean: int):
    plt.figure()
    plt.plot(stock.rolling(window=long_mean).mean(), label="long_mean")
    plt.plot(stock.rolling(window=short_mean).mean(), label="short_mean")
    plt.plot(stock)


def identify_series_crossing(stock: pd.Series, long_mean: int, short_mean: int):
    long_SMA = stock.rolling(window=long_mean).mean().dropna().rename("long_SMA")
    short_SMA = stock.rolling(window=short_mean).mean().iloc[(long_mean-1):].rename("short_SMA")
    new_df = pd.concat([long_SMA, short_SMA], axis=1)

    short_higher = new_df.apply(lambda x: (1 if (x['long_SMA'] < x['short_SMA']) else 0), axis=1)
    short_higher_t = short_higher.iloc[1:].reset_index(drop=True)
    short_higher_tp1 = short_higher.iloc[:-1].reset_index(drop=True)

    new_df = new_df.iloc[1:]
    cross = ((short_higher_t + short_higher_tp1) == 1).rename("cross")
    new_df = pd.concat([new_df, pd.DataFrame(cross).set_index(new_df.index)], axis=1)

    return new_df.loc[new_df['cross']].index.to_list()
