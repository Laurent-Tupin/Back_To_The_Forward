from Data.get_market_data import load_stock_price
from sklearn.metrics import mean_squared_error

airbus_ts = load_stock_price(90, 'AIR.PA')

airbus_close = airbus_ts['4. close']

train = airbus_close.iloc[0:59]
test = airbus_close.iloc[59:89]

# Simple Moving Average Example
sma_pred = train.rolling(window=5).mean()

# TODO CY: add MSE between test and sma_pred
# np.sqrt(mean_squared_error(sma_pred, test)