from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os
import config

ts = TimeSeries(config.key, output_format='pandas')


def load_stock_price(days, ticker, API=True, save=False, outputsize='full'):
    file_name = os.path.join(config.data_dir, f'{ticker}.csv')
    if API:
        stock_ts = ts.get_daily(ticker, outputsize)[0]
        if save:
            stock_ts.to_csv(file_name)
            print(f'{ticker}.csv is saved in {config.data_dir}')
    else:
        if not os.path.exists(file_name):
            raise Exception(f'Market data for {ticker} has not been saved to directory: {config.data_dir} -_-')
        stock_ts = pd.read_csv(file_name)
    return stock_ts.iloc[0:days - 1][::-1]
