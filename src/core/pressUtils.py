'''
anything involving retrieving, analyzing, and cleaning press release data
'''
import MySQLdb
import requests
import logging
import pytz
import threading
import time
import json
import finnhub
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
from core.models import PressReleases
from core.dailyUtils import get_daily_data
from core.sentimentUtils import get_press_sentiment, get_press_toneshift
from core.utils import my_pct_change, cut_decimals, winsor_data, zscore, check_outliers, exp_decay, plot_standardized_data, remove_invalid_characters, z_score_series
from core.dbUtils import ensure_connection

log = logging.getLogger(__name__)

'''
NOTE: variables to be tested
    - winsor level
    - exponential decay level
    - short, medium, long window
process: get initial data, drop unnecessary columns, calculate percent change, winsor data if needed, z-score normalize, store data
'''
# overwrite - boolean to determine if we are willing to change start date to last date in db - ie dont overwrite for realtime data
def get_press_releases(symbol, start_date, end_date, overwrite, underwrite):
    try: 
        # first check if data already in db
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_pressreleases WHERE symbol = '{symbol}';"
        df_db = pd.read_sql(query, conn)

        # if data present and we are overwriting, set start date to last date in db
        if len(df_db.index) != 0:
            # shift timestamps to EST/EDT
            df_db = df_db.set_index('date').sort_index()
            df_db.index = df_db.index.tz_localize('UTC')
            df_db.index = df_db.index.tz_convert('US/Eastern')

            if overwrite:
                # convert datetime to date
                start_date = df_db.index[-1].date() - timedelta(days=50)

            if underwrite:
                end_date = df_db.index.min().date() + timedelta(days=50)
            
        df = get_finn_press_releases(symbol=symbol, start_date=start_date, end_date=end_date)

        if len(df.index) == 0:
            log.debug("press df empty - skipping")
            return

        # make set of urls in db - used to skip press we already have
        db_url = set(df_db['url'])

        for i in range(len(df["sentiment_score"])):
            if df['url'][i] not in db_url:
                log.debug(f"new press release found at {datetime.now(tz=pytz.timezone('US/Eastern')).time()} | press time: {df.index[i].astimezone(pytz.timezone('US/Eastern'))}")
                ensure_connection() # ensure mysql connection
                PressReleases(
                    symbol=symbol,
                    date=df.index[i],

                    headline=df['headline'][i],
                    description=df['description'][i],
                    url=df['url'][i],
                    sentiment_score=df["sentiment_score"][i],
                    toneshift_score=df["toneshift_score"][i],
                    count=df['count'][i],
                ).save()
            else:
                log.debug(f"press entry already in db: {df['headline'][i]} press time: {df.index[i].astimezone(pytz.timezone('US/Eastern'))}")
        
        conn.close()
    except Exception as err:
        log.error(f"error getting finnhub press releases: {err}", exc_info=True)
        return
    
def get_finn_press_releases(symbol, start_date, end_date):
    try:
    
        press = {
            "date": [],
            "headline": [],
            "description": [],
            'url': [],
            "sentiment_score": [],
            "toneshift_score": [],
            "count": [],
        }

        retry_attempts = 5 # incase of 500 or 502 error

        while start_date < end_date:
            client = finnhub.Client(api_key=config.finnhub.apikey)

            to = start_date + timedelta(days=50)

            # get press from finnhub
            try:
                response = client.press_releases(symbol, _from=start_date.strftime("%Y-%m-%d"), to=to.strftime("%Y-%m-%d"))
            except Exception as err:
                time.sleep(10) # give server time
                if retry_attempts <= 0:
                    log.warning(f"press: error making request - skipping: {err}", exc_info=True)
                    start_date = to - timedelta(days=1)
                    retry_attempts = 5 # reset attempts
                    continue
                else:
                    log.warning(f"press: error making request - retry attempts left: {retry_attempts}")
                    retry_attempts -= 1
                    continue
            
            retry_attempts = 5
            time.sleep(float(config.time_buffer))
            results = response.get("majorDevelopment")
            for result in results:
                # get and clean data - save if all data valid
                date = datetime.strptime(result.get("datetime"), "%Y-%m-%d %H:%M:%S")
                date = pytz.timezone('US/Eastern').localize(date) # TODO: verify this
                date = date.astimezone(pytz.timezone('UTC')) # convert to utc

                headline = result.get("headline")
                headline = remove_invalid_characters(text=headline)

                description = json.dumps(result.get("description"))
                description = remove_invalid_characters(text=description)

                url = json.dumps(result.get("url"))

                sentiment_score = get_press_sentiment(symbol=symbol, txt=description)
                toneshift_score = get_press_toneshift(symbol=symbol, txt=description)

                # TODO: get press origin and schedule

                # if sentiment nil - skip
                if sentiment_score != None and toneshift_score != None:
                    press["date"].append(date) 
                    press["headline"].append(headline)
                    press["description"].append(description)
                    press['url'].append(url)
                    press["sentiment_score"].append(sentiment_score)
                    press["toneshift_score"].append(toneshift_score)
                    press["count"].append(1)
                    log.debug(f"Date: {result.get("datetime")} | headline: {result.get("headline")}")

            start_date = to - timedelta(days=1)
        
        df = pd.DataFrame(press, index=press["date"]).sort_index()

        # remove duplicates 
        df = df[~df.duplicated(subset=['url'], keep='first')]

        return df
    except Exception as err:
        log.error(err, exc_info=True)
        return

# TODO: param for party and schedule - only keep first party data - how do we handle scheduled vs unscheduled?
def get_press_from_db(symbol, tradingFrame):
    try:
        # get news from db
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_pressreleases WHERE symbol = '{symbol}';"
        df = pd.read_sql(query, conn)
        if len(df.index) == 0:
            log.warning(f"press db data not availble for {symbol}")
            return None

        # drop any duplicates
        df = df.drop_duplicates(subset=['date', 'headline', 'description'], keep='last')
        log.debug(f"number of press releases: {len(df['headline'])}")

        # shift timestamps to EST/EDT
        df = df.set_index('date')
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        eastern = pytz.timezone('US/Eastern')
        cutoff_date = datetime.combine(config.cutoff_date, datetime.min.time())
        cutoff_date = eastern.localize(cutoff_date)
        log.debug(f"cutoff datetime: {cutoff_date}")
        if df.index.min() > cutoff_date:
            log.warning(f"not enough press release data for {symbol}: start date: {df.index.min()} | cutoff: {cutoff_date}")
            return None
        
        # NOTE: temp code for research paper
        min_press = 20
        if len(df['sentiment_score']) < min_press:
            log.info(f"not enough press releases for {symbol}: {len(df['sentiment_score'])} | required: {min_press}")
            return None

        df = get_press_indicators(df=df, tradingFrame=tradingFrame)

        # df = cut_decimals(df)

        # df_pct = my_pct_change(df)

        # df = winsor_data(df, config.finnhub.press.winsor_level, 1 - config.finnhub.press.winsor_level)
        # df_pct = winsor_data(df_pct, config.finnhub.press.winsor_level, 1 - config.finnhub.press.winsor_level)

        # norm_df = zscore(df)
        # norm_df_pct = zscore(df_pct)

        # norm_df = norm_df.round(2)
        # norm_df_pct = norm_df_pct.round(2)

        # check_outliers(norm_df.iloc[1:-2])
        # check_outliers(norm_df_pct.iloc[1:-2])

        # norm_df = norm_df.replace({np.nan:None})
        # norm_df_pct = norm_df_pct.replace({np.nan:None})
    
        df = df.add_suffix('_press')

        conn.close()
        return df
    except Exception as err:
        log.error(err, exc_info=True)
        return None

# TODO: verify math
def get_press_indicators(df, tradingFrame):
    try:
        df = df.copy()
        long_window = config.finnhub.press.long_window
        medium_window = config.finnhub.press.medium_window
        short_window = config.finnhub.press.short_window

        df = df[df['party'] == "first"]
        df.drop(columns=["headline", "description", "party"], inplace=True)
        log.debug(f"press columns 1: {df.keys()}")

        # NOTE: consider setting this to df.index.min and not config.start_date
        eastern = pytz.timezone('US/Eastern')
        start_time = datetime.combine(config.start_date, datetime.min.time())
        start_time = eastern.localize(start_time)
        current_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))

        # extend based on if trading frame is daily or intraday
        if tradingFrame == 'intraday':
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_hour)
            new_time_index = pd.date_range(start=start_time, end=end_time, freq='H')

            df.index = df.index.ceil("H")
            df = merge_dates(df=df)
        elif tradingFrame == 'daily':
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            current_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_day)
            new_time_index = pd.date_range(start=start_time, end=end_time, freq='D')
            
            # before extending data, round down all timestamps to midnigh
            df.index = df.index.normalize()
            df = merge_dates(df=df)

            # mark when a press occured - should be every index before extension
            press_occurred = [1] * len(df['count'])            
            df['press_occurred'] = pd.Series(press_occurred, index=df.index)

            log.debug(f"press data before extension: {df}")
        else:
            log.error(f"invalid trading frame - cannot get press data")
            return

        df_extended = df.reindex(new_time_index)
        log.debug(f"press: tradingFrame: {tradingFrame} | index: {df_extended.index}")
        df = df_extended.fillna(0)

        # get basic sentiment - volume 
        df['sentiment_volume'] = df['sentiment_score'] * df['count']
        log.debug(f"press columns 2: {df.keys()}")

        df['AS_Long'] = ta.sma(df['sentiment_score'], length=long_window)
        df['AS_Medium'] = ta.sma(df['sentiment_score'], length=medium_window)
        df['AS_Short'] = ta.sma(df['sentiment_score'], length=short_window)

        df['AV_Long'] = ta.sma(df['count'], length=long_window)
        df['AV_Medium'] = ta.sma(df['count'], length=medium_window)
        df['AV_Short'] = ta.sma(df['count'], length=short_window)

        df['SVS_Long'] = df['AS_Long'] * df['AV_Long']
        df['SVS_Medium'] = df['AS_Medium'] * df['AV_Medium']
        df['SVS_Short'] = df['AS_Short'] * df['AV_Short']

        df['ATS_Long'] = ta.sma(df['toneshift_score'], length=long_window)
        df['ATS_Medium'] = ta.sma(df['toneshift_score'], length=medium_window)
        df['ATS_Short'] = ta.sma(df['toneshift_score'], length=short_window)

        # momentum toneshift = cur avg toneshift - prev avg toneshift
        df["Mtm_Toneshift_Long"] = df["ATS_Long"].diff()
        df["Mtm_Toneshift_Medium"] = df["ATS_Medium"].diff()
        df["Mtm_Toneshift_Short"] = df["ATS_Short"].diff()

        # momentum sentiment 
        df["Mtm_Sentiment_Long"] = df["AS_Long"].diff()
        df["Mtm_Sentiment_Medium"] = df["AS_Medium"].diff()
        df["Mtm_Sentiment_Short"] = df["AS_Short"].diff()

        # momentum sentiment-volume
        df["Mtm_SV_Long"] = df["SVS_Long"].diff()
        df["Mtm_SV_Medium"] = df["SVS_Medium"].diff()
        df["Mtm_SV_Short"] = df["SVS_Short"].diff()

        # toneshift sentiment interaction = cur avg toneshift * cur avg sentiment
        df["TSSI_Long"] = df["ATS_Long"] * df["AS_Long"]
        df["TSSI_Medium"] = df["ATS_Medium"] * df["AS_Medium"]
        df["TSSI_Short"] = df["ATS_Short"] * df["AS_Short"]

        # divergence = sentiment - toneshift = indicates change in direction
        df["Divergence_Long"] = df["AS_Long"] - df["ATS_Long"]
        df["Divergence_Medium"] = df["AS_Medium"] - df["ATS_Medium"]
        df["Divergence_Short"] = df["AS_Short"] - df["ATS_Short"]

        df['SVS_Divergence'] = df['SVS_Short'] - df['SVS_Long']

        return df
    except Exception as err:
        log.error(f"error getting press indicators: {err}", exc_info=True)
        return

def merge_dates(df):
    try:
        df = df.copy()

        def classify_press_type(values):
            schedules = set(values)
            if schedules == {'scheduled'}:
                return 'scheduled'
            elif schedules == {'unscheduled'}:
                return 'unscheduled'
            elif schedules == {'announcement'}:
                return 'announcement'
            else:
                return 'mixed'

        schedule_series = df.groupby(df.index)['schedule'].agg(classify_press_type)
        schedule_series.name = "schedule"

        agg_df = df.groupby(df.index).agg({
            "sentiment_score": 'sum',  
            "count": 'sum',
            "toneshift_score": 'sum',
        })

        agg_df['schedule'] = schedule_series

        log.debug(f"merge dates results: {agg_df.head()}")

        return agg_df
    except Exception as err:
        log.error(f"error merging press dates: {err}", exc_info=True)

# for each symbol, get data from the last hour and save to db
def realTimePressData(symbols, date):  
    try:
        for symbol in symbols:
            log.debug(f"getting current press data for {symbol} at {datetime.now().time()}")
            # get todays press data
            start_date = date - timedelta(days=10)
            end_date = date + timedelta(days=1)
            get_press_releases(symbol=symbol, start_date=start_date, end_date=end_date, overwrite=False)
    except Exception as err:
        log.error(f"error getting current press data: {err}", exc_info=True)

# gets data from db but if press occurs after 4pm, it is associated with the next day
def get_stoch_press(symbol):
    try:
        qs = PressReleases.objects.filter(symbol=symbol)
        df= pd.DataFrame.from_records(qs.values())
        df.drop(columns=['uid'], inplace=True)
        df.drop_duplicates(subset=['date', 'headline', 'description'], keep='last', inplace=True)

        # if time after market close, add a day
        df['date'] = df['date'] + np.where(df['date'].dt.hour >= 16, timedelta(days=1), timedelta(0)) # np.where: if time greater then 4pm, add day. otherwise, add 0

        # shift timestamps to EST/EDT
        df = df.set_index('date')
        # df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        eastern = pytz.timezone('US/Eastern')
        cutoff_date = datetime.combine(config.cutoff_date, datetime.min.time())
        cutoff_date = eastern.localize(cutoff_date)
        log.debug(f"cutoff datetime: {cutoff_date}")
        if df.index.min() > cutoff_date:
            log.warning(f"not enough press release data for {symbol}: start date: {df.index.min()} | cutoff: {cutoff_date}")
            return None
        
        min_press = 20
        if len(df['sentiment_score']) < min_press:
            log.info(f"not enough press releases for {symbol}: {len(df['sentiment_score'])} | required: {min_press}")
            return None

        df = get_press_indicators(df=df, tradingFrame='daily')
        df = df.add_suffix('_press')
        return df
    except Exception as err:
        log.error(f"error getting press data for stochastic model: {err}", exc_info=True)
        return None