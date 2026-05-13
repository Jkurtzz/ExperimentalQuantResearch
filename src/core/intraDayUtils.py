import alpaca_trade_api
import pytz
import requests
import logging
import time
import MySQLdb
import websockets
import json
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf

from django.utils import timezone
from core.config import config
from datetime import datetime, timedelta
from core.models import IntraDayData
from core.utils import my_pct_change, cut_decimals, check_outliers, winsor_data, zscore, plot_standardized_data
from core.dbUtils import ensure_connection

log = logging.getLogger(__name__)
# overwrite - boolean to determine if we are willing to change start date to last date in db - ie dont overwrite for realtime data
def get_stock_price_data(symbol, start_date, end_date, timeframe, overwrite):
    try:
        # first check if data already in db
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_intradaydata WHERE symbol = '{symbol}';"
        df_db = pd.read_sql(query, conn)

        # if data present and we are overwriting, set start date to last date in db
        if len(df_db.index) != 0 and overwrite:
            # shift timestamps to EST/EDT
            df_db = df_db.set_index('timestamp').sort_index()
            df_db.index = df_db.index.tz_localize('UTC')
            df_db.index = df_db.index.tz_convert('US/Eastern')
            # convert datetime to date
            start_date = df_db.index[-1].date() - timedelta(days=1)

        df = fetch_alpaca_stock_data(symbol=symbol, start_date=start_date, end_date=end_date, tradingframe=timeframe)
        
        # check if empty df returned
        if df is None:
            return
        if df.empty:
            return
        
        # drop any accidental duplicates
        df = df[~df.index.duplicated(keep='first')]

        # make set of the index in the db - used to skip any data we already have
        db_indexes = set(df_db.index)

        # save data to db
        for i in range(len(df['open'])):
            if df.index[i] not in db_indexes:
                log.info(f"new stock price entry for {symbol} at {df.index[i]}")
                ensure_connection() # ensure mysql connection
                IntraDayData(
                    symbol=symbol,
                    timestamp=df.index[i],

                    # prices
                    open=df["open"][i],
                    high=df["high"][i],
                    low=df["low"][i],
                    close=df["close"][i],
                    volume=df["volume"][i],
                    dollar_volume=df["dollar_volume"][i], 
                ).save()
            else: 
                log.debug(f"stock entry already in db: {df.index[i]} for {symbol}")
        
        conn.close()
        return
    except Exception as err:
        log.error(f"error getting intraday stock data: {err}", exc_info=True)
        return

# makes api calls to alpaca to get historical stock price data
# TODO: speed this up by using the python library
def fetch_alpaca_stock_data(symbol, start_date, end_date, tradingframe):
    try:
        ext_hours = config.alpaca.ext_hours
        retry_attempts = 3
    
        if tradingframe == 'daily':
            timeframe = '1Day'
        elif tradingframe == 'intraday':
            timeframe = '1Hour'
        else:
            log.error(f"invalid timeframe for stock data: {timeframe} - cannot get data")
            return

        stock_prices = {
            'Date': [],
            'open': [], 
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
            'dollar_volume': [],
        }

        while start_date < end_date:
            date = None
            url = 'https://data.alpaca.markets'
            api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

            try:
                response = api.get_bars(symbol=symbol, timeframe=timeframe, feed='sip', sort='asc', start=start_date, end=end_date, limit=10000)
                time.sleep(0.3)
            except Exception as err:
                if retry_attempts > 0:
                    retry_attempts -= 1
                    log.error(f"unable to get stock data - retrying")
                    continue
                else:
                    log.error(f"unable to get stock data for {symbol} - skipping")
                    return

            for result in response:
                date = result.t.to_pydatetime()
                log.debug(f"alpaca stock data date: {date}")

                open = float(result.o)
                high = float(result.h)
                low = float(result.l)
                close = float(result.c)
                volume = int(result.v)

                if any(v is None for v in [date, open, high, low, close, volume]):
                    log.error("response variable is null")
                    return
                
                stock_prices['Date'].append(date)
                stock_prices['open'].append(open)
                stock_prices['high'].append(high)
                stock_prices['low'].append(low)
                stock_prices['close'].append(close)
                stock_prices['volume'].append(volume)
                stock_prices['dollar_volume'].append(volume * close)

            # set start date to final time entry - make sure to strip any times after
            # NOTE: this could cause infinite loop - need to set end_date to tradeable day
            if date:
                start_date = date.date() + timedelta(days=1)
                log.info(f"{symbol} stock data - new start date: {start_date}")
            else:
                log.info(f"date not defined for {symbol} - stock data not available")
                break 
        
        df = pd.DataFrame(stock_prices, index=stock_prices['Date']).sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # return if dataframe is empty - happens if getting realtime data on closed market hours or company not public during timeframe
        if df.empty:
            log.debug("dataframe empty - returning")
            return df
        
        # df.index = df.index.tz_convert('US/Eastern')

        # Filter out extended hours if ext_hours is False
        if not ext_hours and tradingframe == 'intraday': # if we are using pre-market hours for hourly data
            df = df[~((df.index.hour > 16) |  # Hours after 16 (4:00 PM)
                    (df.index.hour < 4) )]
            
        # TEST: convert back to UTC
        df.index = df.index.tz_convert('UTC')

        return df
    except Exception as err:
        log.error(f"error getting intraday data: {err}", exc_info=True)
        return None
    
'''
gets stock prices from db, calculates indicators, and standardizes data
'''
def get_stock_price_data_from_db(symbol, tradingFrame):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_intradaydata WHERE symbol = '{symbol}';"
        df = pd.read_sql(query, conn)

        # fix timestamps
        df = df.set_index('timestamp')
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        # drop unneeded data
        df = df.drop(columns=['uid', 'symbol'])
        log.debug(f"keys: {df.keys()}")

        df = df[~df.index.duplicated(keep='first')]

        df = get_indicators(df)

        # handle very small numbers before getting percent change - this prevents massive percent changes
        df = cut_decimals(df)

        if config.alpaca.ext_hours and tradingFrame == 'intraday':
            # interpolated data for missing extended hours
            start_time = df.index.min()
            # have data end at the most recent hour
            current_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_hour)
            new_time_index = pd.date_range(start=start_time, end=end_time, freq='H')
            
            # extend original and original pct data
            df_hourly = df.reindex(new_time_index)
            df_ip = df_hourly.interpolate(method='linear')
        else:
            # sloppy code but less code to rewrite
            df_ip = df

        # get rate of change and acceleration
        df_roc = get_roc(df_ip)
        df_acc = get_roc(df_roc)

        # get percent change for df, df_roc, and df_acc
        df_pct = my_pct_change(df_ip)
        df_roc_pct = my_pct_change(df_roc)
        df_acc_pct = my_pct_change(df_acc)

        # handle infinities by setting them to 95th and 5th percenttile - winsorization
        df_pct = winsor_data(df_pct, config.alpaca.intra_day.winsor_level, 1 - config.alpaca.intra_day.winsor_level)
        # plot_standardized_data(df=df_pct, title="intra day percent change data")

        df_roc_pct = winsor_data(df_roc_pct, config.alpaca.intra_day.winsor_level, 1 - config.alpaca.intra_day.winsor_level)

        df_acc_pct = winsor_data(df_acc_pct, config.alpaca.intra_day.winsor_level, 1 - config.alpaca.intra_day.winsor_level)

        # z-score normalize data for all dfs
        norm_df = zscore(df_ip)
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

        conn.close()
    
        return df
    except Exception as err:
        log.error(f"error getting intra day stock data: {err}", exc_info=True)
        return


'''
    @param df - data fram containing stock data and indicators'''
def get_indicators(df):
    try:

        bbands = None
        # get sma values
        df['sma_low'] = ta.sma(df['close'], length=config.sma.low_window)
        # df['sma_mid'] = ta.sma(df['close'], length=config.sma.mid_window)
        df['sma_high'] = ta.sma(df['close'], length=config.sma.high_window)
        
        # get ema values
        df['ema_low'] = ta.ema(df['close'], length=config.ema.low_window)
        df['ema_mid'] = ta.ema(df['close'], length=config.ema.mid_window)
        df['ema_high'] = ta.ema(df['close'], length=config.ema.high_window)
            
        # get rsi values
        df['rsi_low'] = ta.rsi(close=df['close'], length=config.rsi.low_window)
        df['rsi_high'] = ta.rsi(close=df['close'], length=config.rsi.high_window)
        
        # get bollinger bands
        bbands = ta.bbands(df['close'], length=config.bollinger.window, std=config.bollinger.std_dev)
        df['lower_band'] = bbands[f'BBL_{config.bollinger.window}_{config.bollinger.std_dev}']
        df['sma_mid'] = bbands[f'BBM_{config.bollinger.window}_{config.bollinger.std_dev}']
        df['upper_band'] = bbands[f'BBU_{config.bollinger.window}_{config.bollinger.std_dev}']
        df['band_width'] = bbands[f'BBB_{config.bollinger.window}_{config.bollinger.std_dev}']
        df['band_percentage'] = bbands[f'BBP_{config.bollinger.window}_{config.bollinger.std_dev}']

        # get macd values
        # need long_ema_window + signal_window amount of prices for macd
        short_ema_low = ta.ema(df['close'], length=config.macd.low.short_window)
        long_ema_low = ta.ema(df['close'], length=config.macd.low.long_window)
        df['macd_line_low'] = short_ema_low - long_ema_low
        df['signal_line_low'] = ta.ema(df['macd_line_low'], length=config.macd.low.signal_window)
        df['histogram_low'] = df['macd_line_low'] - df['signal_line_low']

        # macd calculations
        short_ema_high = ta.ema(df['close'], length=config.macd.high.short_window)
        long_ema_high = ta.ema(df['close'], length=config.macd.high.long_window)
        df['macd_line_high'] = short_ema_high - long_ema_high
        df['signal_line_high'] = ta.ema(df['macd_line_high'], length=config.macd.high.signal_window)
        df['histogram_high'] = df['macd_line_high'] - df['signal_line_high']
        
        # get stochastic oscillators and backfill values - lists dont start at index 0 so we have to backfill with 'None'
        stoch_low = ta.stoch(high=df['high'], low=df['low'], close=df['close'], k=config.stoch.low.k, d=config.stoch.low.d)
        df['stoch_k_low'] = stoch_low[f'STOCHk_{config.stoch.low.k}_{config.stoch.low.d}_3']
        # df['stoch_k_low'] = [None] * (config.stoch.low.k - 1) + list(df['stoch_k_low'])
        df['stoch_d_low'] = stoch_low[f'STOCHd_{config.stoch.low.k}_{config.stoch.low.d}_3']
        # df['stoch_d_low'] = [None] * (config.stoch.low.k - 1) + list(df['stoch_d_low'])

        stoch_high = ta.stoch(high=df['high'], low=df['low'], close=df['close'], k=config.stoch.high.k, d=config.stoch.high.d)
        df['stoch_k_high'] = stoch_high[f'STOCHk_{config.stoch.high.k}_{config.stoch.high.d}_3']
        # df['stoch_k_high'] = [None] * (config.stoch.high.k - 1) + list(df['stoch_k_high'])
        df['stoch_d_high'] = stoch_high[f'STOCHd_{config.stoch.high.k}_{config.stoch.high.d}_3']        
        # df['stoch_d_high'] = [None] * (config.stoch.high.k - 1) + list(df['stoch_d_high'])   

        # get on-balance volume values
        df['obv'] = ta.obv(close=df['close'], volume=df['volume']) # REDO!!!

        # get average true range values
        df['atr'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=config.atr.window)
        
        # get commodity channel index
        df['cci'] = ta.cci(high=df['high'], low=df['low'], close=df['close'], length=config.cci.window)

        # get vwap
        df['vwap'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])

        # get cmf
        df['cmf'] = ta.cmf(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=config.cmf.window)

        # get wpr
        df['wpr'] = ta.willr(high=df['high'], low=df['low'], close=df['close'], length=config.wpr.window)

        # mark pre-market hours
        df['pre_market'] = mark_pre_market_hours(df=df)

        return df
    except Exception as err:
        log.error(f"error getting intraday indicators: {err}", exc_info=True)
        return

# rate of change in 'per hour' time units
def get_roc(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        prev_values = df_copy[col].shift(1)
        time_diff = df_copy.index.to_series().diff().dt.total_seconds() / (60 * 60)  # time diff in hours
        col_diff = df_copy[col] - prev_values 
        roc = col_diff / time_diff
        roc = roc.where(df_copy[col].notna() & prev_values.notna(), np.nan)

        df_copy[col] = roc
    return df_copy


def mark_pre_market_hours(df):
    df = df.copy()
    pre_market = []
    for dt in df.index:
        if (dt.hour < 10 or (dt.hour == 9 and dt.minute < 30)):
            pre_market.append(1)
        else:
            pre_market.append(0)
    return pre_market


async def open_intraday_socket(stocks):
    try:
        async with websockets.connect(config.alpaca.stockSocketUrl) as websocket:
            auth_message = {
                'action': 'auth',
                'key': config.alpaca.apikey,
                'secret': config.alpaca.secret,
            }

            await websocket.send(json.dumps(auth_message))
            response = await websocket.recv()
            log.debug(f"auth response: {response}")

            sub_msg = {
                'action': 'subscribe',
                'bars': stocks
            }
            await websocket.send(json.dumps(sub_msg))

            while True:
                message = await websocket.recv()
                data = json.loads(message)
                log.debug(f"bar received")
                log.debug(data)
    except Exception as err:
        log.error(f"error running stock price webosocket: {err}", exc_info=True)

# for each stock, make api call to get closing price at beginning of each hour
def intraday_real_time(stockSymbols, date):
    try:
        while True:
            # sleep until beginning of next hour
            cur_time = datetime.now()
            next_hour = cur_time.replace(minute=5, second=0, microsecond=0) + timedelta(hours=1)
            log.debug(f"current time: {cur_time} | sleeping until {next_hour}")
            sleep_time = (next_hour - cur_time).total_seconds()
            time.sleep(sleep_time)

            for symbol in stockSymbols:
                log.debug(f"getting current prices for {symbol} at {datetime.now().time()}")
                start_date = date - timedelta(days=1)
                end_date = date + timedelta(days=1)
                get_intra_day_data(symbol=symbol, start_date=start_date, end_date=end_date, overwrite=False)     
                # TODO: get latest stock price and save that to db
                      
    except Exception as err:
        log.error(f"error making proactive intraday api calls: {err}", exc_info=True)

def get_latest_price(symbol):
    try:
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        order = api.get_latest_bar(symbol=symbol, feed='sip')
        log.debug(f"order: {order.c}")
        return order.c
    except Exception as err:
        log.error(f"unable to get current price for {symbol}: {err}", exc_info=True)
    
# gets the bottom of the hour price
def get_boh_price(symbol):
    try:
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        order = api.get_bars(symbol=symbol, timeframe='1H', limit=1, feed='sip', sort='desc')
        log.debug(f"order: {order[-1].c}")
        return order[-1].c
    except Exception as err:
        log.error(f"unable to get current price for {symbol}: {err}", exc_info=True)

def getLatestStockPriceFromDb(symbol):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)

        # get latest entry from db
        query = f"SELECT * FROM core_intradaydata WHERE symbol = '{symbol}' ORDER BY timestamp DESC LIMIT 1;"
        df = pd.read_sql(query, conn)

        # fix timestamps
        df = df.set_index('timestamp')
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        conn.close()
        return df
    except Exception as err:
        log.error(f"unable to get most recent stock price from db: {err}", exc_info=True)
        return None

# gets the latest 5 minute stock prices and returns the sma 5 and 10
def getShortTermSma(symbol, timeframe1, timeframe2):
    try:
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        response = api.get_bars(symbol=symbol, timeframe='5Min', feed='sip', sort='asc')
        prices = []
        
        for result in response:
            prices.append(float(result.c))
        
        prices = pd.Series(prices)

        smaLow = ta.sma(prices, length=timeframe1)
        smaHigh = ta.sma(prices, length=timeframe2)

        # incase early market hours dont have enough prices - if this causes issues, maybe use 1 minute prices
        if smaLow is None or smaHigh is None:
            return 0, 0

        smaLow = float(smaLow.iloc[-1])
        smaHigh = float(smaHigh.iloc[-1])

        return smaLow, smaHigh
    except Exception as err:
        log.error(f"error getting current sma vals: {err}", exc_info=True)
        return 0, 0

# returns boolean if a price dip happened in the last x minutes
def dipOccurred(symbol, pct_drop, timeframe):
    try:
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        response = api.get_bars(symbol=symbol, timeframe='5Min', feed='sip', sort='asc')

        prices = []

        for result in response:
            prices.append(float(result.c))

        prices = pd.Series(prices)

        # get last x entries based on timeframe
        timeframe = int(timeframe / 5) # 5 minute intervals so get last timeframe/5 entries
        prices = prices.tail(timeframe)

        peak = prices.max()
        cur_price = prices.iloc[-1]

        log.debug(f"{timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))} | {symbol} | peak: {peak} | cur price: {cur_price}")

        return (((cur_price - peak) / peak) <= pct_drop)
    except Exception as err:
        log.error(f"error determining if dip occurred: {err}", exc_info=True)
        return False
    
''' determines status of the market for the next day
    helps us determine whether to short or long the next day
    necessary to avoid trading on correction days that could throw off trades

    @param end_date - cut off data for SPY data
                    - date type

    market score based on the following:
        - SPY 3-day % change    - overbought/oversold - most important
        - SPY 1-day % change    - momentum confirmation
        - VIX                   - risk-on vs risk-off 
        - Advance/Decline ratio - breadth of rally
        - ITR/ATR               - risk
        - Sector momentum       - market conviction
    '''
def get_market_condition(end_date):
    try:
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        market_score = 0

        # get 3-day and 1-day SPY % change
        start_date = end_date - timedelta(days=7)
        response = api.get_bars(symbol='SPY', timeframe='1Day', feed='sip', sort='asc', start=start_date, end=end_date)

        prices = {
            "high": [],
            "low": [],
            "close": [],
        }

        for result in response:
            prices['high'].append(float(result.h))
            prices['low'].append(float(result.l))
            prices['close'].append(float(result.c))

        log.debug(f"last price date: {result.t}")
        df_prices = pd.DataFrame(prices)

        latest_price = df_prices['close'].iloc[-1]
        yesterday_price = df_prices['close'].iloc[-2]
        three_day_price = df_prices['close'].iloc[-4]

        pct_change = (latest_price - three_day_price) / three_day_price
        log.debug(f"SPY check: three day price: {three_day_price} | yesterdays price: {yesterday_price} | latest price: {latest_price} | pct change: {pct_change}")

        # return WHAT WE ARE BLOCKING 
        if pct_change > 0.025:
            return "long"
        elif pct_change < -0.025:
            return "short"
        else:
            return "neutral"

        # NOTE: unused code for now - consider using down the line
        # if pct_change > 0.02:
        #     market_score -= 2
        # elif pct_change < -0.02:
        #     market_score += 2

        # pct_change = (latest_price - yesterday_price) / yesterday_price

        # if pct_change > 0.01:
        #     market_score -= 1
        # elif pct_change < -0.01:
        #     market_score += 1

        # # ITR/ATR
        # df_prices['atr'] = ta.atr(high=df_prices['high'], low=df_prices['low'], close=df_prices['close'], length=14)
        # latest_atr = df_prices['atr'][-1]
        # itr = df_prices['high'][-1] / df_prices['low'][-1]

        # if (itr / latest_atr) > 1.2:
        #     market_score -= 1

        # # get VIX
        # vix = yf.Ticker("^VIX")
        # vix_hist = vix.history()
        # log.debug(f"vix data: {vix_hist}")

        # latest_vix = vix_hist['Close'].iloc[-1]
        # log.debug(f"latest vix: {latest_vix}")

        # if latest_vix > 22:
        #     market_score -= 1
        # elif latest_vix < 15:
        #     market_score += 1

        # # advance/decline ratio
        # ad_ratio = get_ad_ratio(retry_attempts=2)
        # if ad_ratio < 0.6: 
        #     market_score -= 1
        # elif ad_ratio > 1.5:
        #     market_score += 1

        # if market_score > 2:
        #     return "bullish"
        # elif market_score < -2:
        #     return "bearish"
        # else:
        #     return "neutral" 
    except Exception as err:
        log.error(f"error getting market condition: {err}", exc_info=True)
        return None

'''
grabs raw data from db without getting indicators
@param symbol - str
'''
def get_stock_price_data_from_db_quick_grab(symbol):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_intradaydata WHERE symbol = '{symbol}';"
        df = pd.read_sql(query, conn)

        if df.empty:
            return

        # fix timestamps
        df = df.set_index('timestamp')
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        # drop unneeded data
        df = df.drop(columns=['uid', 'symbol'])
        log.debug(f"{symbol} stock price keys: {df.keys()}")

        df = df[~df.index.duplicated(keep='first')]

        conn.close()
        return df
    except Exception as err:
        log.error(f"error getting intra day stock data: {err}", exc_info=True)
        return

def get_market_regimes(spy_df):
    # market regime calcs
    spy_sma50 = ta.sma(spy_df['close'], length=50)
    spy_sma200 = ta.sma(spy_df['close'], length=200)
    spy_return_30d = spy_df['close'].pct_change(30)
    regime = []
    for i in spy_df.index:
        if (not np.isnan(spy_sma50[i]) and not np.isnan(spy_sma200[i]) and not np.isnan(spy_return_30d[i])):
            if spy_sma50[i] > spy_sma200[i] and spy_return_30d[i] > 0.03:
                regime.append(1)  
            elif spy_sma50[i] < spy_sma200[i] and spy_return_30d[i] < -0.03:
                regime.append(-1)  
            else:
                regime.append(0)  
        else:
            regime.append(np.nan)
    return pd.Series(regime, index=spy_df.index)

def get_spy_prices():
    api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=config.alpaca.marketUrl, api_version='v2')
    spy_dict = {
        'date': [],
        'close': [],
    }

    try: 
        response = api.get_bars(symbol='SPY', timeframe='1Day', feed='sip', sort='asc', start=config.start_date, end=datetime.today().date(), limit=10000)
    except Exception as err:
        log.error(f"error getting stock price data for SPY")
        return
    for result in response:
        date = result.t.to_pydatetime()
        spy_dict['date'].append(date)
        spy_dict['close'].append(float(result.c))

    spy_df = pd.DataFrame(spy_dict).set_index('date')
    spy_df.index = spy_df.index.tz_convert('US/Eastern')
    return spy_df