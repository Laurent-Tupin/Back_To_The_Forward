import pandas as pd

def calc_pairwise(series: pd.Series()):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    series_t = series[:-1].reset_index(drop=True)
    series_tp1 = series[1:].reset_index(drop=True)
    return series_t, series_tp1