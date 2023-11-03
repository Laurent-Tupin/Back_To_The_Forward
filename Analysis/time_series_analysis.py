import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from datetime import datetime as dt

import config
from utils.time_series_utils import calc_pairwise

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


def find_peak_trough(stock: pd.Series, long_mean: int, short_mean: int, plot_freq: int):
    long_SMA = stock.rolling(window=long_mean).mean().dropna().rename("long_SMA")
    short_SMA = stock.rolling(window=short_mean).mean().iloc[(long_mean - 1):].rename("short_SMA")
    stock_df = pd.concat([stock.iloc[(long_mean - 1):], long_SMA, short_SMA], axis=1)

    # determine when the short SMA series is crossing the long SMA series
    # cross condition is if long_SMA[t] < short_SMA[t] and long_SMA[t+1] > short_SMA[t+1]
    # to avoid using a for loop, create a series to determine when x['long_SMA'] < x['short_SMA'
    # then replicate the series and shift it by one to do pairwise comparison
    short_higher = stock_df.apply(lambda x: (1 if (x['long_SMA'] < x['short_SMA']) else 0), axis=1)
    short_higher_t, short_higher_tp1 = calc_pairwise(short_higher)

    # (short_higher_t + short_higher_tp1) = 0 means long_SMA[t] > short_SMA[t] and long_SMA[t+1] > short_SMA[t+1]
    # (short_higher_t + short_higher_tp1) = 1 means long_SMA[t] > short_SMA[t] and long_SMA[t+1] < short_SMA[t+1]
    # (short_higher_t + short_higher_tp1) = 2 means long_SMA[t] < short_SMA[t] and long_SMA[t+1] < short_SMA[t+1]
    cross = pd.DataFrame(((short_higher_t + short_higher_tp1) == 1).rename("cross")). \
        set_index(stock_df.iloc[1:].index)
    cross_dates = cross.loc[cross['cross']].index.to_list()
    first_date = stock_df.index[0]
    last_date = stock_df.index[-1]
    all_cross_dates = [first_date] + cross_dates + [last_date]
    stock_df['sequence'] = range(0, (stock.size - long_mean + 1))

    # between every two crosses, we for a cluster and for each cluster we find the peak and trough by finding the
    # date with maximum and minimum
    peak_dfs = []
    trough_dfs = []
    for d in range(0, len(all_cross_dates) - 1):
        rows = range(stock_df.loc[all_cross_dates[d], 'sequence'],
                     stock_df.loc[all_cross_dates[d + 1], 'sequence'])

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
                trough_dfs.append(local_df.loc[local_df['sequence'] == min_price_date,
                                               [config.stock_price_col, 'percentage_diff']])
            elif (local_df['percentage_diff'] >= 0).all() & peak_trough_condition(max_price_date):
                peak_dfs.append(local_df.loc[local_df['sequence'] == max_price_date,
                                             [config.stock_price_col, 'percentage_diff']])

    peak_df = pd.concat(peak_dfs)
    trough_df = pd.concat(trough_dfs)
    peak_trough_df = pd.concat([pd.DataFrame(peak_df).assign(feature="peak"),
                                pd.DataFrame(trough_df).assign(feature="trough")]).sort_index()

    # now ensure that a peak is always followed by a trough and vice versa
    # whenever there are more than one peaks or troughs consecutively,
    # we only keep the one with the maximum percentage difference from the long SMA
    # we do this by assigning each consecutive run of peaks or troughs to groups 1,2,3...
    group_pt = 1
    group = [group_pt]
    for pt in range(1, peak_trough_df.shape[0]):
        if peak_trough_df.iloc[pt]['feature'] != peak_trough_df.iloc[pt - 1]['feature']:
            group_pt = group_pt + 1
        group = group + [group_pt]

    peak_trough_df['group'] = group
    unique_pt_df = peak_trough_df.groupby(['group', 'feature'])['percentage_diff'].max()
    # create the final df showing the final set of peaks and troughs along with their dates and price
    final_df = unique_pt_df.reset_index().merge(
        peak_trough_df.reset_index()[['date', 'percentage_diff', config.stock_price_col]],
        how='left'
    ).set_index('date').drop(columns='group')

    plt.figure()
    # plt.plot(stock_df['long_SMA'], label="long_mean")
    plt.plot(stock_df[config.stock_price_col])
    plt.plot(final_df.loc[final_df['feature'] == 'peak'][config.stock_price_col], 'o')
    plt.plot(final_df.loc[final_df['feature'] == 'trough'][config.stock_price_col], 'o')
    plt.xticks(stock_df.index[pd.Series(range(0, stock_df.shape[0] // plot_freq)).multiply(plot_freq)])
    plt.show()

    return final_df



def predict_next_turning_point(df: pd.DataFrame, lag: int):
    index0 = pd.to_datetime(df.index)

    df.index = pd.DatetimeIndex(df.index).to_period('D')
    date_0, date_1 = calc_pairwise(df.index)
    date_diff = date_1.astype('int64') - date_0.astype('int64')

    mod_price = AutoReg(df[config.stock_price_col], lag)
    res_price = mod_price.fit()
    mod_date = AutoReg(date_diff, lag)
    res_date = mod_date.fit()

    starting_point = (df.index[-1] - df.index[0]).n + 1
    ending_point = starting_point
    price_pred = res_price.predict(starting_point, ending_point).reset_index(drop=True)[0]
    date_pred = res_date.predict(starting_point, ending_point).reset_index(drop=True)[0]

    next_date = dt.strftime((pd.DateOffset(days=round(date_pred)) + index0[-1]), "%Y-%m-%d")
    prediction = pd.DataFrame({config.stock_price_col: [price_pred]}, index=[next_date])

    return prediction

