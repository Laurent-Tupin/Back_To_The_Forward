import numpy as np
import pandas as pd
import config
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


def find_peak_trough(stock: pd.Series, long_mean: int, short_mean: int,):

    long_SMA = stock.rolling(window=long_mean).mean().dropna().rename("long_SMA")
    short_SMA = stock.rolling(window=short_mean).mean().iloc[(long_mean - 1):].rename("short_SMA")
    stock_df = pd.concat([stock.iloc[(long_mean - 1):], long_SMA, short_SMA], axis=1)

    short_higher = stock_df.apply(lambda x: (1 if (x['long_SMA'] < x['short_SMA']) else 0), axis=1)
    short_higher_t = short_higher.iloc[1:].reset_index(drop=True)
    short_higher_tp1 = short_higher.iloc[:-1].reset_index(drop=True)

    cross = pd.DataFrame(((short_higher_t + short_higher_tp1) == 1).rename("cross")). \
        set_index(stock_df.iloc[1:].index)
    cross_dates = cross.loc[cross['cross']].index.to_list()

    first_date = stock_df.index[0]
    last_date = stock_df.index[-1]
    all_cross_dates = [first_date] + cross_dates + [last_date]
    stock_df['sequence'] = range(0, (stock.size - long_mean + 1))

    peak_dfs = []
    trough_dfs = []
    for d in range(0, len(all_cross_dates) - 1):
        rows = range(stock_df.loc[all_cross_dates[d], 'sequence'],
                     stock_df.loc[all_cross_dates[d + 1], 'sequence'])

        # TODO CY: place minimum window length in config
        if len(rows) > config.min_peak_trough_window:
            local_df = stock_df.iloc[rows].copy()
            local_df['percentage_diff'] = local_df['short_SMA'].divide(local_df['long_SMA']) - 1
            max_percentage_diff_date = local_df.loc[
                (local_df['percentage_diff'].abs() == local_df['percentage_diff'].abs().max()),
                'sequence'].values[0]
            max_price_date = local_df.loc[
                (local_df[config.stock_price_col] == local_df[config.stock_price_col].max()),
                'sequence'].values[0]
            min_price_date = local_df.loc[
                (local_df[config.stock_price_col] == local_df[config.stock_price_col].min()),
                'sequence'].values[0]

            # TODO CY: find a more solid way to define a bound between maximum percentage_diff and highest/lowest price
            def peak_trough_condition(use_date):
             return abs(max_percentage_diff_date - use_date) < np.round(local_df.shape[0] / 2.)

            if (local_df['percentage_diff'] < 0).all() & peak_trough_condition(min_price_date):
                trough_dfs.append(local_df.loc[local_df['sequence'] == min_price_date, config.stock_price_col])
            elif (local_df['percentage_diff'] >= 0).all() & peak_trough_condition(max_price_date):
                peak_dfs.append(local_df.loc[local_df['sequence'] == max_price_date, config.stock_price_col])

    peak_df = pd.concat(peak_dfs)
    trough_df = pd.concat(trough_dfs)
    final_df = pd.concat([pd.DataFrame(peak_df).assign(feature="peak"),
                          pd.DataFrame(trough_df).assign(feature="trough")]).sort_index()

    plt.figure()
    plt.plot(stock_df['long_SMA'], label="long_mean")
    plt.plot(stock_df['short_SMA'], label="short_mean")
    plt.plot(stock_df[config.stock_price_col])
    plt.plot(peak_df, 'o')
    plt.plot(trough_df, 'o')
    plt.show()

    return final_df




