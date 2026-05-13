'''
anything involving retrieving, analyzing, and cleaning insider transaction data
'''
import MySQLdb
import requests
import logging
import pytz
import threading
import finnhub
import time
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
from core.models import InsiderTransactions
from core.utils import my_pct_change, cut_decimals, winsor_data, zscore, check_outliers, exp_decay, plot_standardized_data, rolling_percentile, z_score_series
from core.dbUtils import ensure_connection


log = logging.getLogger(__name__)
# process: get initial data, drop unnecessary columns, calculate percent change, winsor data if needed, z-score normalize, store data
def get_insider_transactions(symbol, start_date, end_date, overwrite):
    try:
        # first check if data already in db
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_insidertransactions WHERE symbol = '{symbol}';"
        df_db = pd.read_sql(query, conn)

        # if data present and we are overwriting, set start date to last date in db
        if len(df_db.index) != 0 and overwrite:
            # shift timestamps to EST/EDT
            df_db = df_db.set_index('filing_date').sort_index()
            df_db.index = df_db.index.tz_localize('UTC')
            df_db.index = df_db.index.tz_convert('US/Eastern')
            # convert datetime to date
            start_date = df_db.index[-1].date() - timedelta(days=100)

        df = get_finnhub_insider_transactions(symbol=symbol, start_date=start_date, end_date=end_date)

        if len(df.index) == 0:
            log.debug("insider df empty - skipping")
            return

        # make set of ids in db - used to skip any transactions we already saved
        db_ids = set(df_db['id'])

        for i in range(len(df["dollar_volume_change"])):
            # skip any data we already have
            if df['id'][i] not in db_ids: # NOTE: assuming if id not in db, none of the transactions in that filing are
                log.debug(f"new insider transaction found at {datetime.now(tz=pytz.timezone('US/Eastern')).time()} | transaction time: {df.index[i].astimezone(pytz.timezone('US/Eastern'))}")
                ensure_connection() # ensure mysql connection
                InsiderTransactions(
                    symbol=symbol,

                    filing_date=df.index[i],
                    transaction_date=df['transaction_date'][i],

                    id=df['id'][i],
                    volume_change=df["volume_change"][i],
                    transaction_price=df['transaction_price'][i],
                    transaction_code=df['transaction_code'][i],
                    dollar_volume_change=df["dollar_volume_change"][i],
                    transaction_count=df['transaction_count'][i],
                ).save()
            else:
                log.error(f"transaction already in db: {df.index[i]}")

        conn.close()
        return 
    except Exception as err:
        log.error(f"error getting insider transaction data: {err}", exc_info=True)
        return

def get_finnhub_insider_transactions(symbol, start_date, end_date):
    try:

        insider_transactions = {
            "filing_date": [],
            "transaction_date": [],

            "id": [], # id of the transaction - each id for a specific filing - multiple transactions in a filing
            # NOTE: the assumption is that multiple transactions in a filing happen on the same day
            # so if that id is not in the db, none of them are

            "volume_change": [],
            "transaction_price": [],
            "transaction_code": [],
            "dollar_volume_change": [],
            "transaction_count": [],
        }

        retry_attempts = 5 # incase of 500 or 502 error

        while start_date < end_date:
            client = finnhub.Client(api_key=config.finnhub.apikey)

            to = start_date + timedelta(days=20)

            try:
                response = client.stock_insider_transactions(symbol=symbol, _from=start_date.strftime("%Y-%m-%d"), to=to.strftime("%Y-%m-%d"))
            except Exception as err:
                time.sleep(10) # give server time
                if retry_attempts <= 0:
                    log.warning(f"insider: error making request - skipping: {err}", exc_info=True)
                    start_date = to - timedelta(days=1)
                    retry_attempts = 5 # reset attempts
                    continue
                else:
                    log.warning(f"insider: error making request - retry attempts left: {retry_attempts}")
                    retry_attempts -= 1
                    continue
            
            retry_attempts = 5
            time.sleep(float(config.time_buffer))
            results = response.get("data")
            for result in results:
                if (float(result.get("transactionPrice")) != float(0)): # 0 means no buy or sell happened
                    filing_date = result.get("filingDate")
                    # TODO: further research needed on this - go by different date?
                    if filing_date == "" or filing_date == '':
                        log.warning(f"filing date blank for {symbol} - skipping")
                        continue
                    
                    filing_date = datetime.strptime(filing_date, "%Y-%m-%d")
                    filing_date = pytz.timezone('US/Eastern').localize(filing_date) # TODO: verify this
                    filing_date = filing_date.astimezone(pytz.timezone('UTC')) # convert to utc
                    insider_transactions['filing_date'].append(filing_date)

                    transaction_date = datetime.strptime(result.get("transactionDate"), "%Y-%m-%d")
                    insider_transactions['transaction_date'].append(transaction_date) 

                    insider_transactions['id'].append(result.get("id"))

                    volume_change = float(result.get("change"))
                    insider_transactions["volume_change"].append(volume_change)

                    transaction_price = float(result.get("transactionPrice"))
                    insider_transactions["transaction_price"].append(transaction_price)

                    insider_transactions["transaction_code"].append(result.get("transactionCode"))

                    dollar_volume_change = volume_change * transaction_price
                    insider_transactions["dollar_volume_change"].append(dollar_volume_change)

                    insider_transactions["transaction_count"].append(1)

            start_date = to - timedelta(days=1)

        df = pd.DataFrame(insider_transactions, index=insider_transactions["filing_date"]).sort_index()
        return df
    except Exception as err:
        log.error(f"error getting insider transactions: {err}", exc_info=True)
        return
    
def get_insider_from_db(symbol, tradingFrame, transaction_type):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_insidertransactions WHERE symbol = '{symbol}';"
        df = pd.read_sql(query, conn)

        # check if data is null
        if len(df.index) == 0:
            log.info("insider df is empty - data either not retrieved or not available")
            return None, None
        
        # drop duplicate entries
        df = df.drop_duplicates(
            subset=['id', 'transaction_date', 'volume_change', 'transaction_price', 'transaction_code', 'dollar_volume_change'], keep='last'
            ).reset_index(drop=True)

        # shift timestamps to EST/EDT
        df = df.set_index('filing_date')
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        df.drop(columns=["transaction_date", "transaction_price", "transaction_code", "id"], inplace=True)
        df = merge_dates(df)

        # save copy of raw data
        df_raw = df.copy()
        log.debug(f"{symbol} insider data range before extension: {df.index}")
        df = get_insider_indicators(df, tradingFrame, transaction_type)
        if df is None: # not enough data
            log.warning(f"not enough insider transaction data for {symbol} - skipping")
            return None, None 
        
        df = df.add_suffix('_insider')
        log.debug(f"{symbol} final insider date range: {df.index}")

        conn.close()
        return df_raw, df
    except Exception as err:
        log.error(f"error getting insider transactions from db: {err}", exc_info=True)
        return None, None
    
def get_insider_indicators(df, tradingFrame, transaction_type):
    try:
        df = df.copy()
        long_window = config.finnhub.insider.long_window
        medium_window = config.finnhub.insider.medium_window
        short_window = config.finnhub.insider.short_window

        # mark when an insider transaction occured - should be every index before extension
        transaction_occured = [1] * len(df['dollar_volume_change'])
        df['transaction_occured'] = pd.Series(transaction_occured, index=df.index)
        df['transaction_type'] = pd.Series([transaction_type] * len(df.index), index=df.index)

        percentile = config.insiderPressResearch.percentile
        if config.insiderPressResearch.rolling:
            if transaction_type == 'combined': 
                rolling_window = 50

                if len(df['dollar_volume_change']) < rolling_window:
                    log.warning(f"not enough insider data: {len(df['dollar_volume_change'])} | required: {rolling_window}")
                    return None
                
                rolling_percentile_combined = rolling_percentile(series=df['dollar_volume_change'].abs(), window=rolling_window, percentile=percentile)
                log.debug(f"combined insider spike threshold: {rolling_percentile_combined}")
                rolling_avg = df['dollar_volume_change'].abs().rolling(rolling_window).mean()
                rolling_std = df['dollar_volume_change'].abs().rolling(rolling_window).std()

                df['dv_combined_spike'] = (df['dollar_volume_change'].abs() > rolling_percentile_combined).astype(int)
                df['dv_combined_z_score'] = (df['dollar_volume_change'].abs() - rolling_avg) / rolling_std

            elif transaction_type == 'buy':
                rolling_buy_window = 20

                buys = df.loc[df["dollar_volume_change"] > 0, "dollar_volume_change"].copy()
                # if len(buys) < rolling_buy_window + 1:
                if len(buys) < config.insiderPressResearch.min_buys:
                    log.warning(f"not enough insider buys: {len(buys)} | required: {config.insiderPressResearch.min_buys}")
                    return None
                
                buy_mean = buys.shift(1).rolling(rolling_buy_window, min_periods=rolling_buy_window).mean()
                buy_std  = buys.shift(1).rolling(rolling_buy_window, min_periods=rolling_buy_window).std(ddof=0)

                buy_z = (buys - buy_mean) / buy_std
                buy_z = buy_z.replace([np.inf, -np.inf], np.nan)

                df['dv_z_score'] = np.nan
                df.loc[buys.index, "dv_z_score"] = buy_z

                # remove inital entries with no z-score
                df = df.iloc[rolling_buy_window : ]

                n_buys = len(df[df['dollar_volume_change'] > 0]['dollar_volume_change'])
                n_non_nil = df["dv_z_score"].notna().sum()
                log.debug(f"number of insider transactions: {n_buys} | number of dv {transaction_type} recorded z-scores: {n_non_nil}")
            
            elif transaction_type == 'sell':
                rolling_sell_window = 20

                sells = df.loc[df["dollar_volume_change"] < 0, "dollar_volume_change"].abs().copy()
                # if len(sells) < rolling_sell_window + 1:
                if len(sells) < config.insiderPressResearch.min_sells:
                    log.warning(f"not enough insider sells: {len(sells)} | required: {config.insiderPressResearch.min_sells}")
                    return None
                
                sell_mean = sells.shift(1).rolling(rolling_sell_window, min_periods=rolling_sell_window).mean()
                sell_std  = sells.shift(1).rolling(rolling_sell_window, min_periods=rolling_sell_window).std(ddof=0)

                sell_z = (sells - sell_mean) / sell_std
                sell_z = sell_z.replace([np.inf, -np.inf], np.nan)

                df['dv_z_score'] = np.nan
                df.loc[sells.index, "dv_z_score"] = sell_z

                # remove inital entries with no z-score
                df = df.iloc[rolling_sell_window : ]

                n_sells = len(df[df['dollar_volume_change'] < 0]['dollar_volume_change'])
                n_non_nil = df["dv_z_score"].notna().sum()
                log.debug(f"number of insider transactions: {n_sells} | number of dv {transaction_type} recorded z-scores: {n_non_nil}")
        else:
            if transaction_type == 'combined':
                min_transactions = config.insiderPressResearch.min_transactions
                transactions = df['dollar_volume_change'].abs()
                if len(transactions) < min_transactions:
                    log.warning(f"not enough insider data: {len(df['dollar_volume_change'])} | required: {min_transactions}")
                    return None            
                combined_percentile_val = np.percentile(transactions, percentile)
                combined_avg = transactions.mean()
                combined_std = transactions.std()
                # df['dv_spike'] = (transactions > combined_percentile_val).astype(int)
                df['dv_z_score'] = (transactions - combined_avg) / combined_std
                df['dv_spike'] = (df['dv_z_score'] > 1).astype(int)
                df['dv_percentile'] = transactions.rank(pct=True)
                log.debug(f"number of insider transactions: {len(df['transaction_count'])} | number of dv {transaction_type} spikes: {df['dv_spike'].sum()}")
            elif transaction_type == 'buy':
                min_buys = config.insiderPressResearch.min_buys
                buys = df[df['dollar_volume_change'] > 0]['dollar_volume_change']   
                if len(buys) < min_buys:
                    log.warning(f"not enough insider buys: {len(buys)} | required: {min_buys}")
                    return None
                buy_percentile_val = np.percentile(buys, percentile)
                buy_avg = buys.mean()
                buy_std = buys.std()
                # df['dv_spike'] = (buys > buy_percentile_val).astype(int)
                df['dv_z_score'] = (buys - buy_avg) / buy_std
                df['dv_spike'] = (df['dv_z_score'] > 1).astype(int)
                df['dv_percentile'] = buys.rank(pct=True)
                log.debug(f"number of insider transactions: {len(df['transaction_count'])} | number of dv {transaction_type} spikes: {df['dv_spike'].sum()}")
            elif transaction_type == 'sell':
                min_sells = config.insiderPressResearch.min_sells
                sells = df[df['dollar_volume_change'] < 0]['dollar_volume_change'].abs()   
                if len(sells) < min_sells:
                    log.warning(f"not enough insider sells: {len(sells)} | required: {min_sells}")
                    return None
                sell_percentile_val = np.percentile(sells, percentile)
                sell_avg = sells.mean()
                sell_std = sells.std()
                # df['dv_spike'] = (sells > sell_percentile_val).astype(int)
                df['dv_z_score'] = (sells - sell_avg) / sell_std
                df['dv_spike'] = (df['dv_z_score'] > 1).astype(int)
                df['dv_percentile'] = sells.rank(pct=True)
                log.debug(f"number of insider transactions: {len(df['transaction_count'])} | number of dv {transaction_type} spikes: {df['dv_spike'].sum()}")
            else:
                log.error(f"skipping insider-spike data")

        # fill 0's for hours we dont have any data
        start_time = df.index.min()
        current_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))

        # extend based on if trading frame is daily or intraday
        if tradingFrame == 'intraday':
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_hour)
            new_time_index = pd.date_range(start=start_time, end=end_time, freq='H')
        elif tradingFrame == 'daily':
            current_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_day)
            new_time_index = pd.date_range(start=start_time, end=end_time, freq='D')
        else:
            log.error(f"invalid trading frame - cannot get insider data")
            return
        df_extended = df.reindex(new_time_index)
        log.debug(f"insider: tradingFrame: {tradingFrame} | index: {df_extended.index}")
        df = df_extended.fillna(0)

        if len(df['dollar_volume_change']) < long_window:
            log.warning(f"not enough insider data to calculate indicators: {len(df['dollar_volume_change'])} | required: {long_window}")
            return None
        
        # df['advc_short'] = ta.sma(df['dollar_volume_change'], length=short_window)
        # df['advc_medium'] = ta.sma(df['dollar_volume_change'], length=medium_window)
        # df['advc_long'] = ta.sma(df['dollar_volume_change'], length=long_window)
        df['advc_short'] = ta.sma(df['dv_z_score'], length=short_window)
        df['advc_medium'] = ta.sma(df['dv_z_score'], length=medium_window)
        df['advc_long'] = ta.sma(df['dv_z_score'], length=long_window)

        df['av_short'] = ta.sma(df['transaction_count'], length=short_window)
        df['av_medium'] = ta.sma(df['transaction_count'], length=medium_window)
        df['av_long'] = ta.sma(df['transaction_count'], length=long_window)

        df['advs_short'] = df['advc_short'] * df['av_short']
        df['advs_medium'] = df['advc_medium'] * df['av_medium']
        df['advs_long'] = df['advc_long'] * df['av_long']

        # momentum dv change = cur rolling avg dv change - prev rolling avg dv change
        df['mtm_dv_long'] = df["advc_long"].diff()
        df['mtm_dv_medium'] = df["advc_medium"].diff()
        df['mtm_dv_short'] = df["advc_short"].diff()

        df['advc_sma_diff'] = df['advc_short'] - df['advc_long']
        
        # cross above/below 
        prev_sma_short = df['advc_short'].shift(1)
        prev_sma_long = df['advc_long'].shift(1)
        
        # if latest value just crossed above sma4: 1. otherwise, 0
        df['advc_cross_above'] = ((df['advc_short'] > df['advc_long']) & (prev_sma_short <= prev_sma_long)).astype(int)

        # if latest value just crossed below sma4: 1. otherwise, 0
        df['advc_cross_below'] = ((df['advc_short'] < df['advc_long']) & (prev_sma_short >= prev_sma_long)).astype(int)

        # diverging above/below
        std_long = ta.stdev(df['advc_short'], length=long_window)

        df['advc_dvg_above'] = (df['advc_short'] > (df['advc_long'] + std_long)).astype(int)
        df['advc_dvg_below'] = (df['advc_short'] < (df['advc_long'] - std_long)).astype(int)

        return df
    except Exception as err:
        log.error(err, exc_info=True)
        return None
    

def get_insider_blackout_periods(symbol, transaction_type):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_insidertransactions WHERE symbol = '{symbol}';"
        df = pd.read_sql(query, conn)

        if len(df.index) == 0:
            return None 
        
        df.drop_duplicates(
            subset=['id', 'transaction_date', 'volume_change', 'transaction_price', 'transaction_code', 'dollar_volume_change'], keep='last', inplace=True
        )

        df.set_index('filing_date', inplace=True)
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        df.drop(columns=["transaction_date", "transaction_price", "transaction_code", "id"], inplace=True)
        df = merge_dates(df=df)

        df['transaction_type'] = pd.Series([transaction_type] * len(df.index), index=df.index)
        if transaction_type == 'buy':
            df['trade_occurred'] = (df['dollar_volume_change'] > 0).astype(int)
        else:
            df['trade_occurred'] = (df['dollar_volume_change'] < 0).astype(int)

        df = df[df['trade_occurred'] == 1] # filter out opposite transaction type - ex: if analyzing buys, we dont want sales

        # get 95th percentile of blackout periods using rolling window
        window = 20
        if len(df['trade_occurred']) < window:
            return None
        
        delta = df.index.to_series().diff()
        delta_days = delta / pd.Timedelta(days=1) # convert delta to float days

        # get rolling 95th percentile of delta days
        df['q95_blackout_period'] = delta_days.shift(1).rolling(window=20, min_periods=20).quantile(0.95)
        
        # start index when rolling window is complete
        df = df.iloc[window + 1 : ]

        '''
        to get blackout period, we extend the index, 
        iterate through and see how many days since last trade, 
        then take the last x blackout periods and get 95th percentile
        '''
        start_time = df.index.min()
        end_time = df.index.max()
        new_time_index = pd.date_range(start=start_time, end=end_time, freq='D')
        df = df.reindex(new_time_index)
        df['q95_blackout_period'] = df['q95_blackout_period'].ffill()
        df.fillna(0, inplace=True)

        days_since_transaction = []
        cur_blackout_period = 0
        for i in range(len(df.index)):  
            if df['trade_occurred'][i] == 1:
                cur_blackout_period = 0
            days_since_transaction.append(cur_blackout_period)
            cur_blackout_period += 1
        df['days_since_transaction'] = pd.Series(days_since_transaction, index=df.index)
        
        conn.close()
        df = df.add_suffix('_insider')
        return df
    except Exception as err:
        log.error(f"error getting insider blackout periods: {err}", exc_info=True)
        return None
    
def merge_dates(df):
    agg_df = df.groupby(df.index).agg({
        # NOTE: technically volume should be based on id (multiple sub-transactions for one transaction id)
        "volume_change": 'sum',
        "dollar_volume_change": 'sum',
        "transaction_count": 'sum',
    })
    return agg_df

# for each symbol, get data from the last hour and save to db
def realTimeInsiderData(symbols, date):  
    try:
        for symbol in symbols:
            log.debug(f"getting current insider data for {symbol} at {datetime.now().time()}")
            # get todays insider data
            start_date = date - timedelta(days=30)
            end_date = date + timedelta(days=1)
            get_insider_transactions(symbol=symbol, start_date=start_date, end_date=end_date)
    except Exception as err:
        log.error(f"error getting current insider data: {err}", exc_info=True)