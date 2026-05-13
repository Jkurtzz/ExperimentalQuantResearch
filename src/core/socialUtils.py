'''
anything involving retrieving, analyzing, and cleaning social media data
'''
import requests
import logging
import pytz
import threading
import time
import finnhub
import MySQLdb
import pandas as pd
import pandas_ta as ta
import numpy as np


import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from django.http import JsonResponse
from django.utils import timezone
from core.config import config
from datetime import datetime, timedelta
from core.models import SocialMedia
from core.dailyUtils import get_daily_data
from core.sentimentUtils import get_news_sentiment
from core.utils import my_pct_change, cut_decimals, winsor_data, zscore, check_outliers, plot_standardized_data
from core.dbUtils import ensure_connection

log = logging.getLogger(__name__)


# process: get base data, round small numbers, get rate of change and acceleration, get percent change, winsor data, z-score data, store data
# overwrite - boolean to determine if we are willing to change start date to last date in db - ie dont overwrite for realtime data
def get_social_media_data(symbol, start_date, end_date, overwrite):
    try:
        # first check if data already in db
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_socialmedia WHERE symbol = '{symbol}';"
        df_db = pd.read_sql(query, conn)

        if len(df_db.index) != 0:
            # shift timestamps to EST/EDT
            df_db = df_db.set_index('date').sort_index()
            df_db.index = df_db.index.tz_localize('UTC')
            df_db.index = df_db.index.tz_convert('US/Eastern')

        # if data present and we are overwriting, set start date to last date in db
        if len(df_db.index) != 0 and overwrite:
            # convert datetime to date
            start_date = df_db.index[-1].date() - timedelta(days=1)

        df = get_finn_social_media(symbol=symbol, start_date=start_date, end_date=end_date)

        if len(df.index) == 0:
            log.debug("social media df empty - skipping")
            return

        # make set of the index in the db - used to skip any data we already have
        db_indexes = set(df_db.index)

        for i in range(len(df["volume"])):
            if df.index[i] not in db_indexes:
                log.debug(f"new social media found at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).time()} | social time: {df.index[i].astimezone(pytz.timezone('US/Eastern'))}")
                ensure_connection() # ensure mysql connection
                SocialMedia(
                    symbol=symbol, 
                    date=df.index[i],
                    
                    sentiment=df['sentiment'][i],
                    volume=df['volume'][i],
                    sentiment_volume=df['sentiment_volume'][i],
                ).save()
            else: 
                log.debug(f"social media entry already in db: {df.index[i].astimezone(pytz.timezone('US/Eastern'))}")
                
        conn.close()
        return
    except Exception as err:
        log.error(f"error getting social media data: {err}", exc_info=True)
        return


'''
makes api call to finnhub and gets social media sentiment and post volumes given dates provided in config file
@return df - containing all indicators before roc, acc, or percent change

NOTE: finnhub only has data from Mar 2021 - Mar 2022, Jun 2022 - July 2022, Sept 2024 - Now
'''
def get_finn_social_media(symbol, start_date, end_date):
    try:

        social = {
            "date": [],
            "sentiment": [],
            "volume": [],
            "sentiment_volume": [],
        }

        retry_attempts = 5 # incase of 500 or 502 error

        while start_date < end_date:
            client = finnhub.Client(api_key=config.finnhub.apikey)

            to_ = start_date + timedelta(days=5) # increase?
            if (to_ > end_date):
                to_ = end_date + timedelta(days=1)
            
            # get finnhub social media data
            try:
                response = client.stock_social_sentiment(symbol=symbol, _from=start_date.strftime("%Y-%m-%d"), to=to_.strftime("%Y-%m-%d"))
            except Exception as err:
                time.sleep(10) # give server time
                if retry_attempts <= 0:
                    log.warning(f"social: error making request - skipping: {err}", exc_info=True)
                    start_date = to_ - timedelta(days=1)
                    retry_attempts = 5 # reset attempts
                    continue
                else:
                    log.warning(f"social: error making request - retry attempts left: {retry_attempts}")
                    retry_attempts -= 1
                    continue
            
            retry_attempts = 5
            time.sleep(float(config.time_buffer))
            results = response.get("data")

            for result in results:
                date = datetime.strptime(result.get("atTime"), "%Y-%m-%d %H:%M:%S") 
                # localize datetime index to UTC
                date = pytz.timezone('UTC').localize(date) # TODO: verify this
                # fix timezone issue - convert to EST/EDT - will be converted back to UTC when stored in db
                # convert to EST/EDT
                # date = date.astimezone(pytz.timezone('US/Eastern'))

                volume = result.get("mention")
                sentiment = result.get("score")

                social["date"].append(date)
                social["volume"].append(volume)
                social["sentiment"].append(sentiment)
                social["sentiment_volume"].append(sentiment * volume)

            start_date = to_ - timedelta(days=1)

        # TODO: duplicate entries are being saved - fix this
        df = pd.DataFrame(social, index=social["date"]).sort_index()    
        df = df.loc[~df.index.duplicated(keep='first')]    
        return df
    except Exception as err:
        log.error(f"error getting social media from finnhub: {err}", exc_info=True)
        return
    
def get_social_from_db(symbol, tradingFrame):
    try:
        long_window = config.finnhub.social.long_window
        medium_window = config.finnhub.social.medium_window
        short_window = config.finnhub.social.short_window

        # get data from db
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_socialmedia WHERE symbol = '{symbol}';"
        df = pd.read_sql(query, conn)

        # check if data is null
        if len(df.index) == 0:
            log.info("social df is empty - data either not retrieved or not available")

            # TODO: review commented code and determine if we return None or blank df
            # return blank df with proper index - have to do this to prevent inner join error later on
            # start_time = pd.Timestamp(config.start_date).tz_localize('US/Eastern')
            # end_time = pd.Timestamp(datetime.today().date()).tz_localize('US/Eastern')
            # new_time_index = pd.date_range(start=start_time, end=end_time, freq='H')

            # df = pd.DataFrame()
            # df = df.reindex(new_time_index)
            return None

        # drop any duplicates
        df = df.drop_duplicates(subset=['date'], keep='last')
        log.debug(f"number of social media entries: {len(df['sentiment'])}")

        # shift timestamps to EST/EDT
        df = df.set_index('date')
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        log.debug(f"social media keys: {df.keys()}")
        df = df.drop(columns=['uid', 'symbol'])

        # remove duplicates
        df = df.loc[~df.index.duplicated(keep='first')]

        # fill 0's for hours we dont have any data
        # TODO: swap start time to this:
        # social media data can be sparse - so setting start time to first entry in social data - ie, we have less data to work with
        start_time = df.index.min()
        
        # start_time = pd.Timestamp(config.start_date).tz_localize('US/Eastern')
        # have data end at the most recent hour
        current_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))

        # extend based on if trading frame is daily or intraday
        if tradingFrame == 'intraday':
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_hour)
            new_time_index = pd.date_range(start=start_time, end=end_time, freq='H')
        elif tradingFrame == 'daily':
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            current_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_day)
            new_time_index = pd.date_range(start=start_time, end=end_time, freq='D')
            
            # before extending data, round down all timestamps to midnight
            df.index = df.index.normalize()
            df = merge_dates(df=df)
            log.debug(f"social data before extension: {df}")
        else:
            log.error(f"invalid trading frame - cannot get social data")
            return

        df_extended = df.reindex(new_time_index)
        log.debug(f"social: tradingFrame: {tradingFrame} | index: {df_extended.index}")

        df = df_extended.fillna(0)

        # get indicators
        df = calculate_averages(df, short_window=short_window, medium_window=medium_window, long_window=long_window)

        # clean data - add hour to data and interpolate missing data
        # NOTE: adding hour because we get data an hour late
        # df.index = df.index + timedelta(hours=1)

        # handle very small numbers before getting percent change - this prevents massive percent changes
        df = cut_decimals(df)

        # get rate of change and acceleration
        df_roc = get_roc(df)
        df_acc = get_roc(df_roc)

        # get percent change for df, df_roc, and df_acc
        df_pct = my_pct_change(df)
        df_roc_pct = my_pct_change(df_roc)
        df_acc_pct = my_pct_change(df_acc)

        # handle infinities by setting them to 95th and 5th percenttile - winsorization
        # df = winsor_data(df, 0.05, 0.95)
        df_pct = winsor_data(df_pct, config.finnhub.social.winsor_level, 1 - config.finnhub.social.winsor_level)

        # df_roc = winsor_data(df_roc, 0.05, 0.95)
        df_roc_pct = winsor_data(df_roc_pct, config.finnhub.social.winsor_level, 1 - config.finnhub.social.winsor_level)

        # df_acc = winsor_data(df_acc, 0.05, 0.95)
        df_acc_pct = winsor_data(df_acc_pct, config.finnhub.social.winsor_level, 1 - config.finnhub.social.winsor_level)

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

        # plot_standardized_data(df=norm_df, title=f"social media data for {symbol}")
        
        df = df.add_suffix('_social')

        conn.close()
        return df
    except Exception as err:
        log.error(f"error getting social media data from db: {err}", exc_info=True)
        return
    
'''
takes base dataframe with timestamps, sentiment, and volume and calculate advanced indicators
@param df 
@return df with average sentiment, average volume, and sentiment-volume scores
'''
def calculate_averages(df, short_window, medium_window, long_window):
    try:
        additional_data = {
            'as_all': [], # average sentiment
            'as_long': [],
            'as_medium': [],
            'as_short': [],

            'av_all': [], # average volume
            'av_long': [],
            'av_medium': [],
            'av_short': [],

            'svs_all': [], # average sentiment-volume score
            'svs_long': [],
            'svs_medium': [],
            'svs_short': [],

        }

        for i in range(len(df["volume"])):
            # one per hour so 24 minimum
            if (i + 1 >= 24):
                as_all = sum(df["sentiment"][0:i+1]) / (i + 1)
                additional_data["as_all"].append(as_all)

                av_all = sum(df["volume"][0:i+1]) / (i + 1)
                additional_data["av_all"].append(av_all)

                svs_all = sum(df['sentiment_volume'][0:i+1]) / (i + 1)
                additional_data["svs_all"].append(svs_all)
            else:
                additional_data["as_all"].append(None)
                additional_data["av_all"].append(None)
                additional_data["svs_all"].append(None)

            if i + 1 >= long_window:
                as_long = sum(df["sentiment"][i + 1 - long_window : i + 1]) / long_window
                additional_data["as_long"].append(as_long)

                av_long = sum(df["volume"][i + 1 - long_window : i + 1]) / long_window
                additional_data["av_long"].append(av_long)

                svs_long = sum(df["sentiment_volume"][i + 1 - long_window : i + 1]) / long_window
                additional_data["svs_long"].append(svs_long)
            else:
                additional_data["as_long"].append(None)
                additional_data["av_long"].append(None)
                additional_data["svs_long"].append(None)

            if i + 1 >= medium_window:
                as_medium = sum(df["sentiment"][i + 1 - medium_window : i + 1]) / medium_window
                additional_data["as_medium"].append(as_medium)

                av_medium = sum(df["volume"][i + 1 - medium_window : i + 1]) / medium_window
                additional_data["av_medium"].append(av_medium)

                svs_medium = sum(df["sentiment_volume"][i + 1 - medium_window : i + 1]) / medium_window
                additional_data["svs_medium"].append(svs_medium)
            else:
                additional_data["as_medium"].append(None)
                additional_data["av_medium"].append(None)
                additional_data["svs_medium"].append(None)

            if i + 1 >= short_window:
                as_short = sum(df["sentiment"][i + 1 - short_window : i + 1]) / short_window
                additional_data["as_short"].append(as_short)

                av_short = sum(df["volume"][i + 1 - short_window : i + 1]) / short_window
                additional_data["av_short"].append(av_short)

                svs_short = sum(df["sentiment_volume"][i + 1 - short_window : i + 1]) / short_window 
                additional_data["svs_short"].append(svs_short)
            else: 
                additional_data["as_short"].append(None)
                additional_data["av_short"].append(None)
                additional_data["svs_short"].append(None)

        for col, values in additional_data.items():
            df[col] = values

        # momentum sentiment = cur rolling avg sentiment - prev rolling avg sentiment 
        df["mtm_sentiment_all"] = df["as_all"].diff()
        df["mtm_sentiment_long"] = df["as_long"].diff()
        df["mtm_sentiment_medium"] = df["as_medium"].diff()
        df["mtm_sentiment_short"] = df["as_short"].diff()

        # momentum sentiment-volume
        df["Mtm_SV_All"] = df["svs_all"].diff()
        df["Mtm_SV_Long"] = df["svs_long"].diff()
        df["Mtm_SV_Medium"] = df["svs_medium"].diff()
        df["Mtm_SV_Short"] = df["svs_short"].diff()
        return df
    except Exception as err:
        log.error(f"error calculating additional social media indicators: {err}", exc_info=True)
        return


'''
takes dataframe and calculates rate of change (per hour) and acceleration of all fields
@param df - dataframe
@return df with roc and acc fields with endings '_roc' and '_acc'
'''
def get_roc(df):
    try:
        df_copy = df.copy()
        for col in df_copy.columns:
            prev_values = df_copy[col].shift(1)
            roc = (df_copy[col] - prev_values) 
            roc = roc.where(df_copy[col].notna() & prev_values.notna(), np.nan)

            df_copy[col] = roc
        return df_copy
    except Exception:
        log.error("error getting social roc")


# for each symbol, get data from the last hour and save to db
def realTimeSocialData(symbols, date):  
    try:
        for symbol in symbols:
            log.debug(f"getting current social data for {symbol} at {datetime.now().time()}")
            # get todays social data
            start_date = date - timedelta(days=1)
            end_date = date + timedelta(days=1)
            get_social_media_data(symbol=symbol, start_date=start_date, end_date=end_date, overwrite=False)
    except Exception as err:
        log.error(f"error getting current social data: {err}", exc_info=True)

def merge_dates(df):
    try:
        agg_df = df.groupby(df.index).agg({
            "sentiment": 'sum', # NOTE: sum instead of mean?
            "volume": 'sum',
            "sentiment_volume": 'sum',
        })
        return agg_df
    except Exception as err:
        log.error(f"error merging news dates: {err}", exc_info=True)