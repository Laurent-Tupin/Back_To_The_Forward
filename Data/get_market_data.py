from alpha_vantage.timeseries import TimeSeries

# API Key
key = 'HIIPEUKMK60DTO96'
ts = TimeSeries(key, output_format='pandas')

# TODO CY or LT: modify the function so that it can either choose from a local directory or from online
def load_stock_price(days, ticker, outputsize='full'):
    stock_ts = ts.get_daily(ticker, outputsize)[0]
    return stock_ts.iloc[0:days-1][::-1]

