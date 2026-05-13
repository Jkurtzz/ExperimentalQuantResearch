import logging
import math
import re
import MySQLdb
import alpaca_trade_api
from scipy.stats import linregress
from itertools import combinations
import pandas as pd
import numpy as np
import pandas_ta as ta

from core.config import config
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from core.models import TradeStats

log = logging.getLogger(__name__)
def my_pct_change(df):
    df_copy = df.copy()
    pct_change = (df_copy - df_copy.shift(1)) / df_copy.shift(1).abs()
    pct_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    pct_change.fillna(0, inplace=True)  
    return pct_change

# adds bias - probably dont use
# replaces any infinities in a dataframe
# NOTE: sets value to 1e5 assuming that is larger than any value in dataset - larger numbers can throw off results
def replace_inf(df):
    for key, values in df.items():
        for i in range(len(values)):
            if values[i] == float('inf'):
                values[i] = 1e5
            if values[i] == -float('inf'):
                values[i] = -1e5
        df[key] = values
    return df

# gets z-scores of data in dataframe
def zscore(df):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
    return normalized_df

def z_score_series(series, window):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    z_score = (series - rolling_mean) / rolling_std
    if z_score == np.inf or z_score == -np.inf:
        log.debug(f"z_score is inf: {z_score}")
    return z_score.replace([np.inf, -np.inf], np.nan)

# sets small numbers to 0.001 
def cut_decimals(df):
    df_copy = df.copy()
    for key, values in df_copy.items():
        log.debug(key)
        for i in range(len(values)):
            if abs(values[i]) < 0.001:
                if values[i] > 0:
                    values[i] = 0.001
                elif values[i] < 0:
                    values[i] = -0.001
        df_copy[key] = values    
    return df_copy

''' handle infinities by setting them to l-th and u-th percenttile - winsorization
@param lower - lower bound as a decimal
@param upper - upper bound as a decimal
'''
def winsor_data(df, lower, upper):
    try:
        df_copy = df.copy()
        for key in df_copy.keys():
            l = lower
            u = upper
            # limits can still be +-inf or nan so iterate until we get a valid number
            lower_limit = -float('inf')
            upper_limit = float('inf')
            while (lower_limit == -float('inf') or math.isnan(lower_limit)):
                lower_limit = df_copy[key].quantile(l) 
                l += 0.0001
            while (upper_limit == float('inf') or math.isnan(upper_limit)):
                upper_limit = df_copy[key].quantile(u)
                u -= 0.0001
            log.debug(f"lower limit: {l} | upper limit: {u}")
            df_copy[key] = np.where(df_copy[key] < lower_limit, lower_limit, df_copy[key])
            df_copy[key] = np.where(df_copy[key] > upper_limit, upper_limit, df_copy[key])
        return df_copy
    except Exception as err:
        log.warning("error winsoring data - returning original df")
        return df


def log_iqr(df):
    for key, values in df.items():
        minimum = min(values)
        maximum = max(values)
        q1 = values.quantile(0.25)
        med = values.quantile(0.5)
        q3 = values.quantile(0.75)
        iqr = q3 - q1 

        log.debug(f"field: {key} | min: {minimum} | q1: {q1} | median: {med} | q3: {q3} | max: {maximum} | iqr: {iqr}")
    return

def check_outliers(df):
    for key, values in df.items():
        minimum = min(values)
        maximum = max(values)
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1 
        if (minimum < (q1 - 1.5*iqr)):
            log.debug(f"lower bound outlier for field: {key} | value: {minimum}")
        if (maximum > (q3 + 1.5*iqr)):
            log.debug(f"upper bound outlier for field: {key} | value: {maximum}")
    return


# takes dataframe with extended indices and applies exponential decay to existing values for new entries
def exp_decay(df, rate):
    df_copy = df.copy()
    for key in df_copy.columns:
        # forward fill missing entries with last known entry
        forward_filled = df_copy[key].fillna(method='ffill')
        filled_values = forward_filled.copy()
        for i in range(len(df_copy[key])):
            # if entry is nan, apply exponential decay to last known entry
            if pd.isna(df_copy[key].iloc[i]):
                # find the last known index before or up to the current index
                last_known_indices = np.where(~df_copy[key][:i].isna())[0]
                if last_known_indices.size > 0:
                    last_known_idx = last_known_indices.max()
                    time_diff = i - last_known_idx
                    # apply exponential decay for the missing value
                    filled_values.iloc[i] = forward_filled.iloc[last_known_idx] * np.exp(-rate * time_diff)
                else:
                    # if we are in the beginning of the column and have not found a known value yet
                    filled_values.iloc[i] = np.nan
        df_copy[key] = filled_values
    # plot_standardized_data(df=df_copy, title="exp decay - check")
    return df_copy

def plot_standardized_data(df, title):
    fig = go.Figure()
    for key, values in df.items():
        fig.add_trace(go.Scatter(x=df.index, y=values, mode='lines', name=key))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Z-Score',
        xaxis_rangeslider_visible=True, 
    )

    fig.show()
    return

def histogram(title, rt_vals, observed_val, p_value):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rt_vals, nbinsx=30, name='Random Hit Rates'))

    fig.add_trace(go.Scatter(
        x=[observed_val, observed_val],
        y=[0, 1],  # Will be scaled to match the y-axis
        mode='lines',
        line=dict(color='red', dash='dash'),
        name=f'Observed: {observed_val:.2f}'
    ))

    fig.update_layout(
        title=f'Histogram of Randomized Hit Rates for {title} <br>Observed = {observed_val:.2f}, p = {p_value:.4f}',
        xaxis_title='Hit Rate',
        yaxis_title='Frequency'
    )
    fig.show()

def box_n_whisker(full_title, title1, group1, title2, group2):
    fig = go.Figure()

    fig.add_trace(go.Box(y=group1, name=title1, boxpoints='all', jitter=0.3, pointpos=-1.8))
    fig.add_trace(go.Box(y=group2, name=title2, boxpoints='all', jitter=0.3, pointpos=-1.8))

    fig.update_layout(title=f"Box and Whisker Plot for {full_title} {title1} total: {len(group1)} {title2} total {len(group2)}", yaxis_title="Value")
    fig.show()

# fxn to remove any invalid chars from a string
def remove_invalid_characters(text):
    # remove all non-BMP characters (4-byte Unicode)
    return re.sub(r'[\U00010000-\U0010FFFF]', '', text)


# iterates through trades saved in db, makes api calls to alapca and determine success rates for each trade
def get_trade_stats():
    # get trades from db
    conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
    query = f"SELECT * FROM core_tradestats where success IS NULL;"
    df = pd.read_sql(query, conn)

    if len(df.index) == 0:
        log.info("trade stats has no new entries")
        return
    
    url = config.alpaca.tradeUrl
    api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

    # for each entry, determine if success, failure, or neither (neither stop loss or take profit hit - leave null)
    for i in range(len(df['date'])):
        success = None
        stop_loss_id = df['stop_loss_id'][i]
        take_profit_id = df['take_profit_id'][i]
        if api.get_order(stop_loss_id).status == 'filled':
            success = False
        elif api.get_order(take_profit_id).status == 'filled':
            success = True
        
        log.info(f"success status for {df['symbol'][i]} - {df['order_id'][i]} order at {df['date'][i]}: {success}")

        # save to db
        uid = df['uid'][i]
        trade_obj = TradeStats.objects.get(uid=uid)
        trade_obj.success = success
        trade_obj.save()

        conn.close()
    return

def slope_series(series, window=3):
    slopes = [None] * (window - 1)  # NaN for first few values
    for i in range(window - 1, len(series)):
        y = series[i - window + 1:i + 1]
        x = range(window)
        slope = linregress(x, y).slope
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def get_5_num_summary(vals):
    if len(vals) == 0:
        return None
    min_val = float(np.min(vals))
    q1 = float(np.percentile(vals, 25))
    median = float(np.median(vals))
    q3 = float(np.percentile(vals, 75))
    max_val = float(np.max(vals))

    return (min_val, q1, median, q3, max_val)

def rolling_percentile(series, window, percentile):
    return series.rolling(window).apply(lambda x: np.percentile(x, percentile) if len(x) == window else np.nan)

def get_combinations(lst):
    result = []
    for r in range(1, len(lst) + 1):
        result.extend(list(comb) for comb in combinations(lst, r))
    return result

def get_normalized_return(closing_prices, high_prices, low_prices, window):
    pct_changes = (closing_prices - closing_prices.shift(window)) / closing_prices.shift(window)
    atr = ta.atr(high=high_prices, low=low_prices, close=closing_prices, length=window-1)
    atr = atr.shift(1) # needs to represent past atr's
    normalized_return = pct_changes / atr
    return pd.Series(normalized_return, index=closing_prices.index)

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False