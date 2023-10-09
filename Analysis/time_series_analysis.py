import numpy as np
import pandas as pd
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
