from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np

key = 'HIIPEUKMK60DTO96'
ts = TimeSeries(key, output_format='pandas')


airbus_ts, meta = ts.get_daily('AIR.PA', outputsize='full')
print(airbus_ts)


