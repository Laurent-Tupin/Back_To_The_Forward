import numpy as np
import pandas as pd
from Data.get_market_data import load_stock_price
from sklearn.metrics import mean_squared_error

length_train = 360
stock_price_col = '4. close'


def SMA_window_with_lowest_MSE(ticker, stock_price_col, length_train, min_window_length, n):
    """ Find window length with lowest MSE by looping over n different lengths of windows

    Parameters:
        ticker: ticker of the stock
        stock_price_col: open, high, low or close
        length_train: length of the training set
        min_window_length: minimum length of window
        n: size of loop


    Returns:
        int: length of window
        float: MSE
    """

    stock = load_stock_price(length_train, ticker)[stock_price_col]
    train = stock.iloc[0:(length_train - 1)]
    MSE_list = []
    for i in range(min_window_length, n):
        window = range((length_train - i), (length_train - 1))
        test = stock.iloc[window]
        sma_pred = train.rolling(window=i).mean()
        MSE = np.sqrt(mean_squared_error(sma_pred.iloc[window], test))
        MSE_list = MSE_list + [MSE]

    # TODO CY or LT: how can we write this in a neater way?
    MSE_df = pd.DataFrame({'window_length': range(min_window_length, n), 'MSE': MSE_list}, index=None)
    min_MSE = MSE_df['MSE'] == np.min(MSE_df['MSE'])

    return int(MSE_df.loc[min_MSE, 'window_length'].iloc[0]), np.min(MSE_df['MSE'])


# Airbus example
airbus_window, airbus_MSE = SMA_window_with_lowest_MSE('AIR.PA', stock_price_col, length_train, 10, 59)