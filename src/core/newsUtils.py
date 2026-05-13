'''
anything involving retrieving, analyzing, and cleaning news data
'''
from concurrent.futures import ThreadPoolExecutor
import finnhub.exceptions
import requests
import MySQLdb
import logging
import pytz
import threading
import time
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
from datetime import datetime, timedelta, time as dt_time
from core.models import FinnArticles
from core.dailyUtils import get_daily_data
from core.intraDayUtils import getLatestStockPriceFromDb
from core.utils import my_pct_change, cut_decimals, winsor_data, zscore, check_outliers, plot_standardized_data, remove_invalid_characters
from core.sentimentUtils import get_news_sentiment
from core.dbUtils import ensure_connection

log = logging.getLogger(__name__)

# makes api calls to finnhub and alpaca - stores data in db
def get_news_data(symbol, start_date, end_date):
    try:
        # first check if data already in db
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_finnarticles WHERE symbol = '{symbol}';"
        df_db = pd.read_sql(query, conn)

        # if data present and we are overwriting, set start date to last date in db
        if len(df_db.index) != 0:
            # shift timestamps to EST/EDT
            df_db = df_db.set_index('date').sort_index()
            df_db.index = df_db.index.tz_localize('UTC')
            df_db.index = df_db.index.tz_convert('US/Eastern')
            # convert datetime to date
            start_date = df_db.index[-1].date() - timedelta(days=1)

        df = get_articles(symbol=symbol, start_date=start_date, end_date=end_date)

        if len(df.index) == 0:
            log.debug("news df empty - skipping")
            return

        # make set of urls in db - used to skip articles we already have
        db_urls = set(df_db['url'])

        for i in range(len(df["headline"])):
            log.debug(f"check news index: {df.index[i]}")
            # skip if row is in db or if datetime invalid
            if df['url'][i] not in db_urls and df.index[i].year > 1900:
                log.debug(f"new news article found at {datetime.now(tz=pytz.timezone('US/Eastern')).time()} | article time: {df.index[i].astimezone(pytz.timezone('US/Eastern'))}")
                ensure_connection() # ensure mysql connection
                FinnArticles(
                    symbol=symbol,
                    date=df.index[i],
                    
                    headline=df['headline'][i],
                    summary=df['summary'][i],
                    source=df['source'][i],
                    url=df['url'][i],
                    sentiment=df['sentiment'][i],
                    count=df['count'][i],
                ).save()
            else:
                log.debug(f"news article already in db or invalid: {df['headline'][i]} | {df.index[i]}")
        conn.close()
    except ConnectionError as err:
        log.error(f"connection ended server-side - restarting connection")
        get_news_data(symbol=symbol, start_date=start_date, end_date=end_date)
    except Exception as err:
        log.error(f"error getting news data: {err}", exc_info=True)
    return

'''
    get historical stock news for specific symbol
'''
def get_articles(symbol, start_date, end_date):
    try:
        articles = {
            "Date": [],
            "headline": [],
            "summary": [],
            "source": [],
            "url": [],
            "sentiment": [],
            'count': [],
        }

        retry_attempts = 5 # incase of 500 or 502 error

        while start_date <= end_date:
            client = finnhub.Client(api_key=config.finnhub.apikey)
            log.debug(f"start date: {start_date}")
            to_ = start_date + timedelta(days=1)
            log.debug(f"to: {to_}")

            # get finnhub news
            timer = time.time() 
            try:
                results = client.company_news(symbol, _from=start_date.strftime("%Y-%m-%d"), to=start_date.strftime("%Y-%m-%d")) # finnhub does inclusive ranges so start and end date should be the same
            except Exception as err:
                time.sleep(10) # give server time
                if retry_attempts <= 0:
                    log.warning(f"error making finnhub news request - skipping: {err}", exc_info=True)
                    start_date = to_
                    retry_attempts = 5 # reset attempts
                    continue
                else:
                    log.warning(f"error making finnhub news request - retry attempts left: {retry_attempts}")
                    retry_attempts -= 1
                    continue
            
            time.sleep(float(config.time_buffer))
            for result in results:
                date = datetime.fromtimestamp(result.get("datetime"), tz=pytz.UTC)
                headline = result.get("headline")
                summary = result.get("summary")
                source = result.get("source")
                url = result.get("url")

                if (summary == "Looking for stock market analysis and research with proves results? Zacks.com offers in-depth financial research with over 30years of proven results.") or summary == '':
                    # summary is either blank or invalid from zacks - either skip or set summary equal to headline 
                    log.debug("zacks data or blank summary")
                    if config.finnhub.news.summary_only:
                        continue 
                    else:
                        summary = headline

                # remove any invalid chars from the text
                headline = remove_invalid_characters(text=headline)
                summary = remove_invalid_characters(text=summary)

                sentiment_score = get_news_sentiment(symbol=symbol, txt=summary)
                time.sleep(float(config.time_buffer))
                if (sentiment_score != None):
                    articles["Date"].append(date) 
                    articles["headline"].append(headline) 
                    articles["summary"].append(summary) 
                    articles["source"].append(source) 
                    articles["url"].append(url) 
                    articles["sentiment"].append(sentiment_score)
                    articles['count'].append(1)
                    log.debug(f"Date: {date} | headline: {headline} | source: {source}")
                else:
                    log.warning(f"sentiment score null for \'{headline}\'")

            # get alpaca news
            url = f"{config.alpaca.newsUrl}start={start_date.strftime("%Y-%m-%d")}&end={to_.strftime("%Y-%m-%d")}&sort=desc&symbols={symbol}&limit=50&include_content=true" # start date inclusive, end date exclusive

            headers = {
                "accept": "application/json",
                "APCA-API-KEY-ID": config.alpaca.apikey,
                "APCA-API-SECRET-KEY": config.alpaca.secret
            }

            try:
                response = requests.get(url, headers=headers)
            except Exception as err:
                time.sleep(10) # give server time
                if retry_attempts <= 0:
                    log.warning(f"error making alpaca news request - skipping: {err}", exc_info=True)
                    start_date = to_
                    retry_attempts = 5 # reset attempts
                    continue
                else:
                    log.warning(f"error making alpaca news request - retry attempts left: {retry_attempts}")
                    retry_attempts -= 1
                    continue

            time.sleep(float(config.time_buffer))

            if response == None:
                log.warning(f"null response - skipping: {start_date}")
                start_date = to_
                continue

            if response.status_code != 200:
                log.info(f"status code {response.status_code} - skipping: {start_date}")
                start_date = to_
                continue

            results = response.json().get("news")
            if not results:
                log.error(f"response is empty for api call from {start_date} to {to_}")
                start_date = to_
                continue

            for res in results:
                # get and clean data - save if valid
                date = datetime.strptime(res.get("created_at"), "%Y-%m-%dT%H:%M:%SZ")
                date = pytz.timezone('UTC').localize(date) # TODO: verify this
                # date = date.astimezone(pytz.timezone('US/Eastern')) # fix timezone issue - convert to EST/EDT - will be converted back to UTC when stored in db
                headline = res.get("headline")
                summary = res.get("summary")
                source = res.get("source")
                url = res.get("url")

                # summary can be blank - set to headline
                if summary == '':
                    if config.finnhub.news.summary_only:
                        continue
                    else:
                        summary = headline

                headline = remove_invalid_characters(text=headline)
                summary = remove_invalid_characters(text=summary)

                sentiment_score = get_news_sentiment(symbol=symbol, txt=summary)
                time.sleep(float(config.time_buffer))
                if (sentiment_score != None):
                    articles["Date"].append(date) 
                    articles["headline"].append(headline) 
                    articles["summary"].append(summary) 
                    articles["source"].append(source) 
                    articles["url"].append(url) 
                    articles['sentiment'].append(sentiment_score)
                    articles['count'].append(1)
                    log.debug(f"Date: {date} | headline: {headline} | source: {source}")
                else:
                    log.warning(f"sentiment score null for \'{headline}\'")

            retry_attempts = 5
            start_date = to_
            log.debug(f"time to make api call: {(time.time() - timer)} seconds")

        # create dataframe and clean data - remove duplicates
        df = pd.DataFrame(articles)

        # remove duplicates 
        df = df[~df.duplicated(subset=['summary', 'headline', 'Date'], keep='first')]

        # set datetime as index
        df = df.set_index("Date").sort_index()

        return df
    except Exception as err:
        log.error(f"error getting news from api: {err}", exc_info=True)
        return


'''
gets news articles from db, gets indicators, and cleans data
'''
def get_news_from_db(symbol, tradingFrame):
    try:
        # get news from db
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_finnarticles WHERE symbol = '{symbol}';"
        df = pd.read_sql(query, conn)

        # drop any duplicates
        df = df.drop_duplicates(subset=['date', 'headline', 'source'], keep='last')
        log.debug(f"number of articles: {len(df['sentiment'])}")

        # shift timestamps to EST/EDT
        df = df.set_index('date')
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        log.debug(f"news keys: {df.keys()}")
        # get indicators
        df = get_indicators(df=df, tradingFrame=tradingFrame)

        log.debug(f"sum of articles : {sum(df['count'])}")

        # handle very small numbers before getting percent change - this prevents massive percent changes
        df = cut_decimals(df)

        # add latency to index 
        df.index = df.index + timedelta(hours=config.finnhub.news.latency)

        # get rate of change and acceleration
        df_roc = get_news_roc(df)
        df_acc = get_news_roc(df_roc)

        # get percent change for df, df_roc, and df_acc
        df_pct = my_pct_change(df)
        df_roc_pct = my_pct_change(df_roc)
        df_acc_pct = my_pct_change(df_acc)

        # handle infinities by setting them to kth and 1-kth percenttile - winsorization
        df_pct = winsor_data(df_pct, config.finnhub.news.winsor_level, 1 - config.finnhub.news.winsor_level)

        df_roc_pct = winsor_data(df_roc_pct, config.finnhub.news.winsor_level, 1 - config.finnhub.news.winsor_level)

        df_acc_pct = winsor_data(df_acc_pct, config.finnhub.news.winsor_level, 1 - config.finnhub.news.winsor_level)

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

        # plot_standardized_data(df=norm_df, title="news data")
        df = df.add_suffix('_news')

        conn.close()
        return df
    except Exception as err:
        log.error(f"error getting news data: {err}", exc_info=True)
        return

'''
returns indicators on all news data we have
'''
def get_market_news_from_db(tradingFrame):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_finnarticles;"
        df = pd.read_sql(query, conn)
        df.drop_duplicates(subset=['url'], keep='last', inplace=True)

        if len(df.index) != 0:
            df = df.set_index('date').sort_index()
            df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('US/Eastern')
        else:
            log.warning(f"no market news data available - returning")
            return

        df.drop(columns=['uid', 'symbol', 'headline', 'summary', 'source', 'url'], inplace=True)
        log.debug(f"market news columns: {df.columns.tolist()}")

        if tradingFrame == 'weekly':
            df = df.resample('W-FRI').agg({
                'sentiment': 'mean',
                'count': 'sum',
            })
            log.debug(f"market news new index: {df.index}")
        else:
            log.error(f"unimplemented tradingFrame for get market news from db")
            return 
        
        # indicators
        long = config.finnhub.news.long_window
        medium = config.finnhub.news.medium_window
        short = config.finnhub.news.short_window

        df['sentiment_volume'] = df['sentiment'] * df['count']

        df['sentiment_diff'] = df['sentiment'].diff()
        # df['count_diff'] = df['count'].diff()
        df['sentiment_volume_diff'] = df['sentiment_volume'].diff()

        df['AS_Long'] = ta.ema(df['sentiment'], length=long)
        df['AS_Medium'] = ta.ema(df['sentiment'], length=medium)
        df['AS_Short'] = ta.ema(df['sentiment'], length=short)
        df['AS_Long_diff'] = df['sentiment'] - df['AS_Long']
        df['AS_Medium_diff'] = df['sentiment'] - df['AS_Medium']
        df['AS_Short_diff'] = df['sentiment'] - df['AS_Short']
        
        prev_sentiment = df['sentiment'].shift(1)
        prev_as_long = df['AS_Long'].shift(1)
        prev_as_medium = df['AS_Medium'].shift(1)
        prev_as_short = df['AS_Short'].shift(1)
        df['AS_Long_cross_above'] = ((df['sentiment'] > df['AS_Long']) & (prev_sentiment <= prev_as_long)).astype(int)
        df['AS_Medium_cross_above'] = ((df['sentiment'] > df['AS_Medium']) & (prev_sentiment <= prev_as_medium)).astype(int)
        df['AS_Short_cross_above'] = ((df['sentiment'] > df['AS_Short']) & (prev_sentiment <= prev_as_short)).astype(int)

        df['AS_Long_cross_below'] = ((df['sentiment'] < df['AS_Long']) & (prev_sentiment >= prev_as_long)).astype(int)
        df['AS_Medium_cross_below'] = ((df['sentiment'] < df['AS_Medium']) & (prev_sentiment >= prev_as_medium)).astype(int)
        df['AS_Short_cross_below'] = ((df['sentiment'] < df['AS_Short']) & (prev_sentiment >= prev_as_short)).astype(int)

        df['sentiment_std_long'] = ta.stdev(df['sentiment'], length=long)
        df['sentiment_std_medium'] = ta.stdev(df['sentiment'], length=medium)
        df['sentiment_std_short'] = ta.stdev(df['sentiment'], length=short)
        df['AS_Long_dvg_above'] = (df['sentiment'] > (df['AS_Long'] + df['sentiment_std_long'])).astype(int)
        df['AS_Medium_dvg_above'] = (df['sentiment'] > (df['AS_Medium'] + df['sentiment_std_medium'])).astype(int)
        df['AS_Short_dvg_above'] = (df['sentiment'] > (df['AS_Short'] + df['sentiment_std_short'])).astype(int)
        df['AS_Long_dvg_below'] = (df['sentiment'] < (df['AS_Long'] - df['sentiment_std_long'])).astype(int)
        df['AS_Medium_dvg_below'] = (df['sentiment'] < (df['AS_Medium'] - df['sentiment_std_medium'])).astype(int)
        df['AS_Short_dvg_below'] = (df['sentiment'] < (df['AS_Short'] - df['sentiment_std_short'])).astype(int)

        df['AV_Long'] = ta.ema(df['count'], length=long)
        df['AV_Medium'] = ta.ema(df['count'], length=medium)
        df['AV_Short'] = ta.ema(df['count'], length=short)
        # df['AV_Long_diff'] = df['count'] - df['AV_Long']
        # df['AV_Medium_diff'] = df['count'] - df['AV_Medium']
        # df['AV_Short_diff'] = df['count'] - df['AV_Short']
        
        # prev_volume = df['count'].shift(1)
        # prev_AV_long = df['AV_Long'].shift(1)
        # prev_AV_medium = df['AV_Medium'].shift(1)
        # prev_AV_short = df['AV_Short'].shift(1)
        # df['AV_Long_cross_above'] = ((df['count'] > df['AV_Long']) & (prev_volume <= prev_AV_long)).astype(int)
        # df['AV_Medium_cross_above'] = ((df['count'] > df['AV_Medium']) & (prev_volume <= prev_AV_medium)).astype(int)
        # df['AV_Short_cross_above'] = ((df['count'] > df['AV_Short']) & (prev_volume <= prev_AV_short)).astype(int)

        # df['AV_Long_cross_below'] = ((df['count'] < df['AV_Long']) & (prev_volume >= prev_AV_long)).astype(int)
        # df['AV_Medium_cross_below'] = ((df['count'] < df['AV_Medium']) & (prev_volume >= prev_AV_medium)).astype(int)
        # df['AV_Short_cross_below'] = ((df['count'] < df['AV_Short']) & (prev_volume >= prev_AV_short)).astype(int)

        df['volume_std_long'] = ta.stdev(df['count'], length=long)
        df['volume_std_medium'] = ta.stdev(df['count'], length=medium)
        df['volume_std_short'] = ta.stdev(df['count'], length=short)
        # df['AV_Long_dvg_above'] = (df['count'] > (df['AV_Long'] + df['volume_std_long'])).astype(int)
        # df['AV_Medium_dvg_above'] = (df['count'] > (df['AV_Medium'] + df['volume_std_medium'])).astype(int)
        # df['AV_Short_dvg_above'] = (df['count'] > (df['AV_Short'] + df['volume_std_short'])).astype(int)
        # df['AV_Long_dvg_below'] = (df['count'] < (df['AV_Long'] - df['volume_std_long'])).astype(int)
        # df['AV_Medium_dvg_below'] = (df['count'] < (df['AV_Medium'] - df['volume_std_medium'])).astype(int)
        # df['AV_Short_dvg_below'] = (df['count'] < (df['AV_Short'] - df['volume_std_short'])).astype(int)

        df['SVS_Long'] = df['AS_Long'] * df['AV_Long']
        df['SVS_Medium'] = df['AS_Medium'] * df['AV_Medium']
        df['SVS_Short'] = df['AS_Short'] * df['AV_Short']
        df['SVS_Long_diff'] = df['sentiment_volume'] - df['SVS_Long']
        df['SVS_Medium_diff'] = df['sentiment_volume'] - df['SVS_Medium']
        df['SVS_Short_diff'] = df['sentiment_volume'] - df['SVS_Short']
        
        prev_sentiment_volume = df['sentiment_volume'].shift(1)
        prev_SVS_long = df['SVS_Long'].shift(1)
        prev_SVS_medium = df['SVS_Medium'].shift(1)
        prev_SVS_short = df['SVS_Short'].shift(1)
        df['SVS_Long_cross_above'] = ((df['sentiment_volume'] > df['SVS_Long']) & (prev_sentiment_volume <= prev_SVS_long)).astype(int)
        df['SVS_Medium_cross_above'] = ((df['sentiment_volume'] > df['SVS_Medium']) & (prev_sentiment_volume <= prev_SVS_medium)).astype(int)
        df['SVS_Short_cross_above'] = ((df['sentiment_volume'] > df['SVS_Short']) & (prev_sentiment_volume <= prev_SVS_short)).astype(int)

        df['SVS_Long_cross_below'] = ((df['sentiment_volume'] < df['SVS_Long']) & (prev_sentiment_volume >= prev_SVS_long)).astype(int)
        df['SVS_Medium_cross_below'] = ((df['sentiment_volume'] < df['SVS_Medium']) & (prev_sentiment_volume >= prev_SVS_medium)).astype(int)
        df['SVS_Short_cross_below'] = ((df['sentiment_volume'] < df['SVS_Short']) & (prev_sentiment_volume >= prev_SVS_short)).astype(int)

        df['sentiment_volume_std_long'] = ta.stdev(df['sentiment_volume'], length=long)
        df['sentiment_volume_std_medium'] = ta.stdev(df['sentiment_volume'], length=medium)
        df['sentiment_volume_std_short'] = ta.stdev(df['sentiment_volume'], length=short)
        df['SVS_Long_dvg_above'] = (df['sentiment_volume'] > (df['SVS_Long'] + df['sentiment_volume_std_long'])).astype(int)
        df['SVS_Medium_dvg_above'] = (df['sentiment_volume'] > (df['SVS_Medium'] + df['sentiment_volume_std_medium'])).astype(int)
        df['SVS_Short_dvg_above'] = (df['sentiment_volume'] > (df['SVS_Short'] + df['sentiment_volume_std_short'])).astype(int)
        df['SVS_Long_dvg_below'] = (df['sentiment_volume'] < (df['SVS_Long'] - df['sentiment_volume_std_long'])).astype(int)
        df['SVS_Medium_dvg_below'] = (df['sentiment_volume'] < (df['SVS_Medium'] - df['sentiment_volume_std_medium'])).astype(int)
        df['SVS_Short_dvg_below'] = (df['sentiment_volume'] < (df['SVS_Short'] - df['sentiment_volume_std_short'])).astype(int)

        df.drop(columns=['sentiment_std_long', 'sentiment_std_medium', 'sentiment_std_short', 'volume_std_long', 'volume_std_medium', 'volume_std_short', 'sentiment_volume_std_long', 'sentiment_volume_std_medium', 'sentiment_volume_std_short'], inplace=True)

        # momentum sentiment = cur rolling avg sentiment - prev rollling avg sentiment
        df["Mtm_Sentiment_Long"] = df["AS_Long"].diff()
        df["Mtm_Sentiment_Medium"] = df["AS_Medium"].diff()
        df["Mtm_Sentiment_Short"] = df["AS_Short"].diff()

        # momentum sentiment-volume
        df["Mtm_SV_Long"] = df["SVS_Long"].diff()
        df["Mtm_SV_Medium"] = df["SVS_Medium"].diff()
        df["Mtm_SV_Short"] = df["SVS_Short"].diff()

        # TEST: remove volume
        df.drop(columns=['count', 'AV_Long', 'AV_Medium', 'AV_Short'], inplace=True)
        
        df = df.add_suffix('_news')
        return df
    except Exception as err:
        log.error(f"error getting market news from db: {err}", exc_info=True)
        return

'''
takes dataframe of news articles, gets sentiment, merges timestamps, and get indicators
'''
def get_indicators(df, tradingFrame):
    try:
        long_window = config.finnhub.news.long_window
        medium_window = config.finnhub.news.medium_window
        short_window = config.finnhub.news.short_window 

        # dont modify inplace
        df = df.copy()

        # round to nearest hour and merge timestamps
        df = round_up_time(df=df)
        df = df.drop(columns=['uid', 'symbol', 'headline', 'summary', 'source', 'url'])
        df = merge_dates(df=df)

        # fill 0's for hours we dont have any data
        start_time = df.index.min()
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
            log.debug(f"news data before extension: {df}")
        else:
            log.error(f"invalid trading frame - cannot get news data")
            return

        df_extended = df.reindex(new_time_index)
        log.debug(f"news: tradingFrame: {tradingFrame} | index: {df_extended.index}")

        df = df_extended.fillna(0)

        # get basic sentiment-volume
        df['sentiment_volume'] = df['sentiment'] * df['count']

        indicators = {
            "AS_All": [],
            "AS_Long": [],
            "AS_Medium": [],
            "AS_Short": [],

            "APD_All": [],
            "APD_Long": [],
            "APD_Medium": [],
            "APD_Short": [],

            "SVS_All": [],
            "SVS_Long": [],
            "SVS_Medium": [],
            "SVS_Short": [],
        }

        # sliding window
        day_index = 0        

        for i in range(len(df["sentiment"])):

            # 24 hour counts
            while (df.index[i] - df.index[day_index] > timedelta(days=1)):
                day_index += 1

            # check if len of list is less than a day
            if (df.index[i] - df.index[0]) >= timedelta(days=1):

                all_days = (df.index[i] - df.index[0]).total_seconds() / (24 * 60 * 60)
                apd_all = (sum(df["count"][0:i+1])) / all_days
                indicators["APD_All"].append(apd_all)

                slice_all = df["sentiment"][0:i+1]
                as_all = sum(slice_all) / (i + 1)
                indicators["AS_All"].append(as_all)

                sv_all = sum(df['sentiment_volume'][0:i+1]) / (i + 1)
                indicators["SVS_All"].append(sv_all)
            else: 
                indicators["APD_All"].append(None)
                indicators["AS_All"].append(None)
                indicators["SVS_All"].append(None)
                
    
            # long window
            if (i + 1 >= long_window):
                slice_long = df["sentiment"][i + 1 - long_window : i + 1]
                as_long = sum(slice_long) / long_window
                indicators["AS_Long"].append(as_long)

                long_days = (df.index[i] - df.index[i + 1 - long_window]).total_seconds() / (24 * 60 * 60)
                apd_long = sum(df["count"][i + 1 - long_window : i + 1]) / long_days
                indicators["APD_Long"].append(apd_long)

                sv_long = sum(df["sentiment_volume"][i + 1 - long_window : i + 1]) / long_window
                indicators["SVS_Long"].append(sv_long)
            else:
                indicators["AS_Long"].append(None)
                indicators["APD_Long"].append(None)
                indicators["SVS_Long"].append(None)
            
            # medium window
            if (i + 1 >= medium_window):
                slice_medium = df["sentiment"][i + 1 - medium_window : i + 1]
                as_medium = sum(slice_medium) / medium_window
                indicators["AS_Medium"].append(as_medium)

                medium_days = (df.index[i] - df.index[i + 1 - medium_window]).total_seconds() / (24 * 60 * 60)
                apd_medium = sum(df["count"][i + 1 - medium_window : i + 1]) / medium_days
                indicators["APD_Medium"].append(apd_medium)

                sv_medium = sum(df["sentiment_volume"][i + 1 - medium_window : i + 1]) / medium_window
                indicators["SVS_Medium"].append(sv_medium)
            else:
                indicators["AS_Medium"].append(None)
                indicators["APD_Medium"].append(None)
                indicators["SVS_Medium"].append(None)

            # short window
            if (i + 1 >= short_window):
                slice_short = df["sentiment"][i + 1 - short_window : i + 1]
                as_short = sum(slice_short) / short_window
                indicators["AS_Short"].append(as_short)

                short_days = (df.index[i] - df.index[i + 1 - short_window]).total_seconds() / (24 * 60 * 60)
                apd_short = sum(df["count"][i + 1 - short_window : i + 1]) / short_days
                indicators["APD_Short"].append(apd_short)

                sv_short = sum(df["sentiment_volume"][i + 1 - short_window : i + 1]) / short_window
                indicators["SVS_Short"].append(sv_short)
            else:
                indicators["AS_Short"].append(None)
                indicators["APD_Short"].append(None)
                indicators["SVS_Short"].append(None)

        for col, values in indicators.items():
            df[col] = values

        # momentum sentiment = cur rolling avg sentiment - prev rollling avg sentiment
        df["Mtm_Sentiment_All"] = df["AS_All"].diff()
        df["Mtm_Sentiment_Long"] = df["AS_Long"].diff()
        df["Mtm_Sentiment_Medium"] = df["AS_Medium"].diff()
        df["Mtm_Sentiment_Short"] = df["AS_Short"].diff()

        # momentum sentiment-volume
        df["Mtm_SV_All"] = df["SVS_All"].diff()
        df["Mtm_SV_Long"] = df["SVS_Long"].diff()
        df["Mtm_SV_Medium"] = df["SVS_Medium"].diff()
        df["Mtm_SV_Short"] = df["SVS_Short"].diff()
        return df
    except Exception as err:
        log.error(f"error getting news indicators: {err}", exc_info=True)
        return

# division not required since all data is 1 hour apart
def get_news_roc(df):
    try:
        df_copy = df.copy()
        for col in df_copy.columns:
            prev_values = df_copy[col].shift(1)
            roc = (df_copy[col] - prev_values) 
            roc = roc.where(df_copy[col].notna() & prev_values.notna(), np.nan)

            df_copy[col] = roc
        return df_copy
    except Exception as err:
        log.error(f"error getting news roc: {err}", exc_info=True)

# 12:01 to 13:00 = 13:00
def round_up_time(df):
    try:
        df_copy = df.copy()
        new_idx = []
        for i in df_copy.index:
            if i.minute == 0 and i.second == 0:
                round_time = i  
            else:
                round_time = i + timedelta(minutes=(60 - i.minute), seconds=-i.second)
            new_idx.append(round_time)
        df_copy.index = new_idx
        return  df_copy
    except Exception as err:
        log.error(f"error rounding up news time: {err}", exc_info=True)

def merge_dates(df):
    try:
        agg_df = df.groupby(df.index).agg({
            "sentiment": 'sum', # NOTE: sum instead of mean?
            "count": 'sum',
        })
        return agg_df
    except Exception as err:
        log.error(f"error merging news dates: {err}", exc_info=True)


