import requests
import logging
import pytz
import threading
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from django.http import JsonResponse
from core.config import config
from datetime import datetime, timedelta
from core.models import DailyData
from core.insiderUtils import get_insider_transactions
from core.utils import my_pct_change, cut_decimals, check_outliers, winsor_data, zscore

log = logging.getLogger(__name__)


# process: get base data, drop non-numerical fields, cut small numbers, get rate of change, get acceleration, get percent change, winsor data, z-score normalize, save data
def get_daily_data():
    try:
        start_date = config.start_date
        end_date = config.end_date
        df = get_alpaca_daily_data(start_date=start_date, end_date=end_date)
        plot_prices(df=df)
        df = df.drop(columns=["Date"])
        df = cut_decimals(df)

        pivot_df = df.drop(columns=["Open", "Close", "High", "Low", "Volume"])
        df = df.drop(columns=["P", "R1", "R2", "R3", "S1", "S2", "S3"])
        df_roc = get_roc(df)
        df_acc = get_roc(df_roc)

        # get percent change for df, df_roc, and df_acc
        df_pct = my_pct_change(df)
        df_roc_pct = my_pct_change(df_roc)
        df_acc_pct = my_pct_change(df_acc)

        # handle infinities by setting them to 95th and 5th percenttile - winsorization
        # df = winsor_data(df, 0.05, 0.95)
        df_pct = winsor_data(df_pct, config.alpaca.daily.winsor_level, 1 - config.alpaca.daily.winsor_level)

        # df_roc = winsor_data(df_roc, 0.05, 0.95)
        df_roc_pct = winsor_data(df_roc_pct, config.alpaca.daily.winsor_level, 1 - config.alpaca.daily.winsor_level)

        # df_acc = winsor_data(df_acc, 0.05, 0.95)
        df_acc_pct = winsor_data(df_acc_pct, config.alpaca.daily.winsor_level, 1 - config.alpaca.daily.winsor_level)

        # z-score normalize data for all dfs
        norm_df = zscore(df)
        norm_df_pct = zscore(df_pct)

        norm_df_roc = zscore(df_roc)
        norm_df_roc_pct = zscore(df_roc_pct)

        norm_df_acc = zscore(df_acc)
        norm_df_acc_pct = zscore(df_acc_pct)

        # round final data before storing for easier reading
        norm_df = norm_df.round(2)
        norm_df_pct = norm_df_pct.round(2)

        norm_df_roc = norm_df_roc.round(2)
        norm_df_roc_pct = norm_df_roc_pct.round(2)

        norm_df_acc = norm_df_acc.round(2)
        norm_df_acc_pct = norm_df_acc_pct.round(2)

        # check for outliers before saving data
        check_outliers(norm_df.iloc[1:-2])
        check_outliers(norm_df_pct.iloc[1:-2])

        check_outliers(norm_df_roc.iloc[1:-2])
        check_outliers(norm_df_roc_pct.iloc[1:-2])

        check_outliers(norm_df_acc.iloc[1:-2])
        check_outliers(norm_df_acc_pct.iloc[1:-2])

        # fix the None issue - None doesnt get converted correctly when turned into df
        norm_df = norm_df.replace({np.nan: None})
        norm_df_pct = norm_df_pct.replace({np.nan: None})

        norm_df_roc = norm_df_roc.replace({np.nan: None})
        norm_df_roc_pct = norm_df_roc_pct.replace({np.nan: None})

        norm_df_acc = norm_df_acc.replace({np.nan: None})
        norm_df_acc_pct = norm_df_acc_pct.replace({np.nan: None})

        plot_prices(df=norm_df)

        for i in range(len(norm_df['Open'])):
            DailyData(
                symbol=config.symbol,
                date=norm_df.index[i],

                open=norm_df['Open'][i],
                high=norm_df['High'][i],
                low=norm_df['Low'][i], 
                close=norm_df['Close'][i],
                volume=norm_df['Volume'][i],
                dollar_volume=norm_df["Dollar_volume"][i],

                open_roc=norm_df_roc['Open'][i],
                high_roc=norm_df_roc['High'][i],
                low_roc=norm_df_roc['Low'][i], 
                close_roc=norm_df_roc['Close'][i],
                volume_roc=norm_df_roc['Volume'][i],
                dollar_volume_roc=norm_df_roc["Dollar_volume"][i],

                open_acc=norm_df_acc['Open'][i],
                high_acc=norm_df_acc['High'][i],
                low_acc=norm_df_acc['Low'][i], 
                close_acc=norm_df_acc['Close'][i],
                volume_acc=norm_df_acc['Volume'][i],
                dollar_volume_acc=norm_df_acc["Dollar_volume"][i],

                # percent changes 
                open_pct=norm_df_pct['Open'][i],
                high_pct=norm_df_pct['High'][i],
                low_pct=norm_df_pct['Low'][i], 
                close_pct=norm_df_pct['Close'][i],
                volume_pct=norm_df_pct['Volume'][i],
                dollar_volume_pct=norm_df_pct["Dollar_volume"][i],

                open_roc_pct=norm_df_roc_pct['Open'][i],
                high_roc_pct=norm_df_roc_pct['High'][i],
                low_roc_pct=norm_df_roc_pct['Low'][i], 
                close_roc_pct=norm_df_roc_pct['Close'][i],
                volume_roc_pct=norm_df_roc_pct['Volume'][i],
                dollar_volume_roc_pct=norm_df_roc_pct["Dollar_volume"][i],

                open_acc_pct=norm_df_acc_pct['Open'][i],
                high_acc_pct=norm_df_acc_pct['High'][i],
                low_acc_pct=norm_df_acc_pct['Low'][i], 
                close_acc_pct=norm_df_acc_pct['Close'][i],
                volume_acc_pct=norm_df_acc_pct['Volume'][i],
                dollar_volume_acc_pct=norm_df_acc_pct["Dollar_volume"][i],
                
                # pivot=df_pct['P'][i],
                # R1=df_pct['R1'][i],
                # R2=df_pct['R2'][i],
                # R3=df_pct['R3'][i],
                # S1=df_pct['S1'][i],
                # S2=df_pct['S2'][i],
                # S3=df_pct['S3'][i],
            ).save()
    except Exception as err:
        log.error(f"error getting daily data: {err}")
    return


def get_alpaca_daily_data(start_date, end_date):
    try:
        baseurl = config.alpaca.baseurl
        timeframe = "1Day"
        apikey = config.alpaca.apikey
        secret = config.alpaca.secret
        symbol = config.symbol

        pivot_levels = {
            'Date': [],
            'Open': [], 
            'High': [],
            'Low':[],
            'Close': [],
            "Volume": [],
            "Dollar_volume": [],
            'P': [],
            'R1': [],
            'R2': [],
            'R3': [],
            'S1': [],
            'S2': [],
            'S3': [],
        }

        request = f"{baseurl}/{symbol}/bars?timeframe={timeframe}&start={start_date}&end={end_date}&limit=10000"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": apikey,
            "APCA-API-SECRET-KEY": secret,
        }
        response = requests.get(request, headers=headers)

        if response.status_code != 200:
            log.error(f"failed to fetch daily stock prices: {response.text}")
            return
        
        data = response.json()
        results = data["bars"]
        if len(results) == 0:
            log.error("error getting api reponse: results empty")
            return
        
        for result in results:
            date_str = result['t']
            date_str = date_str.rstrip('Z')
            date = datetime.fromisoformat(date_str)

            open = float(result['o'])
            high = float(result['h'])
            low = float(result['l'])
            close = float(result['c'])
            volume = int(result['v'])

            if any(v is None for v in [date, open, high, low, close, volume]):
                log.error("response variable is null")
                return
            
            # get pivot and res/sup levels for each day
            P, R1, R2, R3, S1, S2, S3 = calculate_levels(high=high, low=low, close=close)
            pivot_levels['Date'].append(date) # when using pivot levels for model training, use the previous days levels
            pivot_levels['Open'].append(open)
            pivot_levels['High'].append(high)
            pivot_levels['Low'].append(low)
            pivot_levels['Close'].append(close)
            pivot_levels['Volume'].append(volume)
            pivot_levels["Dollar_volume"].append(close * volume)
            pivot_levels['P'].append(P)
            pivot_levels['R1'].append(R1)
            pivot_levels['R2'].append(R2)
            pivot_levels['R3'].append(R3)
            pivot_levels['S1'].append(S1)
            pivot_levels['S2'].append(S2)
            pivot_levels['S3'].append(S3)

        df = pd.DataFrame(pivot_levels, index=pivot_levels['Date']).sort_index() # soemthing wrong with date not being a string
        # set to previous day data
        df.index = df.index - timedelta(hours=24)
        return df
    
    except Exception as err:
        log.error(f"error retrieving daily stock data: {err}")
    
    return


'''
    helper function for get_support_resist to calculate the pivot points for each day
    @param high, low, close - daily high, low, and closing prices of a given stock
    @return P, R1, R2, R3, S1, S2, S3 - corresponding levels for each day'''
def calculate_levels(high, low, close):
    P = (high + low + close) / 3
    R1 = (2 * P) - low
    R2 = P + (high - low)
    R3 = high + 2 * (P - low)
    S1 = (2 * P) - high 
    S2 = P - (high - low)
    S3 = low - 2 * (high - P)
    return P, R1, R2, R3, S1, S2, S3


'''
    @param df - data fram containing stock data and indicators'''
def plot_prices(df):  
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=df.index, y=df['High'], mode='lines', name='High'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Low'], mode='lines', name='Low'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    # fig.add_trace(go.Scatter(x=df.index, y=df['P'], mode='lines', name='Pivot'))
    # fig.add_trace(go.Scatter(x=df.index, y=df['R1'], mode='lines', name='Resistance 1'))
    # fig.add_trace(go.Scatter(x=df.index, y=df['R2'], mode='lines', name='Resistance 2'))
    # fig.add_trace(go.Scatter(x=df.index, y=df['R3'], mode='lines', name='Resistance 3'))
    # fig.add_trace(go.Scatter(x=df.index, y=df['S1'], mode='lines', name='Support 1'))
    # fig.add_trace(go.Scatter(x=df.index, y=df['S2'], mode='lines', name='Support 2'))
    # fig.add_trace(go.Scatter(x=df.index, y=df['S3'], mode='lines', name='Support 3'))

    fig.update_layout(
        title='Daily Stock Price with Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True, 
    )

    fig.show()
    return

# rate of change in 'per day'
def get_roc(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        prev_values = df_copy[col].shift(1)
        roc = (df_copy[col] - prev_values) 
        roc = roc.where(df_copy[col].notna() & prev_values.notna(), np.nan)

        df_copy[col] = roc
    return df_copy