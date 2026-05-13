import MySQLdb
import alpaca_trade_api
import joblib
import requests
import logging
import random
import pytz
import time
import threading
import finnhub
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from django.http import JsonResponse
from core.config import config
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
from django.utils import timezone
from core.models import AnnualMacroData, MonthlyMacroData, QuarterlyMacroData, LongStocks, ShortStocks, StockMetrics
from sklearn.preprocessing import StandardScaler
from core.dbUtils import ensure_connection
from core.macroUtils import get_macro, get_macro_from_db
from core.intraDayUtils import get_latest_price, get_stock_price_data, get_stock_price_data_from_db, get_stock_price_data_from_db_quick_grab
from core.earningsUtils import get_earnings, get_earnings_from_db
from core.insiderUtils import get_insider_transactions, get_insider_from_db
from core.utils import my_pct_change, plot_standardized_data, slope_series
from core.trainingUtils import seq_train_model, get_best_feature_combination, train_model
from core.newsUtils import get_market_news_from_db
from core.models import QEarnings, ADRatios

log = logging.getLogger(__name__)

'''chooses stock symbols based on market status, 
    and performance of stocks based on the following metrics:   
        - prices movements
        - beta
        - rsi
        - insider transaction data
        - earnings reports
    @param minCap - minimum market cap for each stock
    @param retry_attempts - retry attempts in case api calls fail
    @param end_date - end date for which we want to get data for
                    - used for testing when we simulate picking stocks for each simulated week
                    - datetime type
'''
def get_symbols(minCap, end_date, retry_attempts):
    try:
        # first determine market status
        market_status = get_market_status(cutoff_date=end_date)
        log.info(f"market status on {end_date}: {market_status}")

        # make api call to get most symbols in us
        client = finnhub.Client(api_key=config.finnhub.apikey)
        mics = ['XNYS','XASE','BATS','XNAS']
        symbols = []
        for mic in mics:
            # get list of symbols
            try:
                response = client.stock_symbols(exchange='US', mic=mic, currency='USD')
            except Exception:
                if retry_attempts > 0:
                    log.warning(f"error getting list of symbols - retry attempts left: {retry_attempts}")
                    time.sleep(10)
                    return get_symbols(minCap=minCap, end_date=end_date, retry_attempts=retry_attempts-1)
                else:
                    log.error(f"error: unable to get symbols")
                    return
                
            for stock in response:
                symbols.append(stock.get('symbol'))

        # remove any stocks with periods - these are substocks
        symbols = list(filter(lambda x: "." not in x, symbols))

        # filter out stocks with below 10 billion
        log.debug(f"number of symbols: {len(symbols)}")

        # without multithreading, it would take 60 hours to analyze data
        i = 0
        x = max(1, len(symbols) // 20) # 1/20 of the symbols
        threads = []
        while i < len(symbols):
            batch = symbols[i:i+x]
            thread = threading.Thread(target=analyze_symbols, args=(batch, market_status, minCap, end_date))
            thread.start()
            threads.append(thread)
            i += x

        for t in threads:
            t.join()

        return market_status
    except Exception as err:
        log.error(f"error getting stock symbols: {err}", exc_info=True)
        return 



'''
@param symbols - list
@param market_status - str
@param minCap - int
@param cutoff_date - datetime
'''
def choose_short_symbols(symbols, market_status, minCap, cutoff_date):
    try:
        log.info("choosing short stocks")
        start_time = time.time()

        # variables depending on market status
        status_vars = {
            'bullish': {
                'rsiLow': -100, # no limit
                'rsiHigh': 50, 
                'minBeta': 0.5,
                'maxBeta': 1.2,
            }, 
            'neutral': {
                'rsiLow': 40,
                'rsiHigh': 60,
                'minBeta': 0.8,
                'maxBeta': 1.2,
            },
            'bearish': {
                'rsiLow': 35,
                'rsiHigh': 60,
                'minBeta': 0.0,
                'maxBeta': 0.7,
            }
        }

        # dictionary used to save symbols
        symbols_dict = {
            'symbol': [],
            'beta': [],
            'market_cap': [],
        }

        '''
        for each stock:
            - get sma, rsi, and beta
            - if all above threshold, get insider data
            - if rolling dollar volume is pointing downward, get quarterly earnings
            - if quarterly earnings are profitable and greater than rolling average, and market cap is above threshold, save symbol
        '''
        for symbol in symbols:
            # slow down api calls per minute
            time.sleep(0.3)

            # first determine if the stock is even shortable
            url = config.alpaca.tradeUrl
            api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

            # put api call in try/catch incase it is not available
            try:
                asset = api.get_asset(symbol=symbol)
            except Exception:
                log.warning(f"asset not available - skipping: {symbol}")
                continue

            if not asset.shortable:
                log.info(f"{symbol} not shortable - skipping")
                continue
                
            # computing beta manually based on date
            beta = compute_beta(symbol=symbol, end_date=cutoff_date)

            # if sma10 less than sma50, save. otherwise, skip
            smaLow = get_sma(symbol=symbol, days=10, end_date=cutoff_date.replace(tzinfo=None).date())
            smaHigh = get_sma(symbol=symbol, days=50, end_date=cutoff_date.replace(tzinfo=None).date())

            # rsi based on market status
            rsi = get_rsi(symbol=symbol, days=14, end_date=cutoff_date.replace(tzinfo=None).date())

            log.debug(f"short: {symbol} smaLow: {smaLow} | smaHigh: {smaHigh} | rsi: {rsi} | beta {beta}")

            # some stocks missing data - skip
            if not beta or not smaLow or not smaHigh or not rsi:
                continue

            # beta greater than 1, market cap greater than 10 billion 10000000000, sma10 > sma50, and 55 <= rsi <= 70
            rsiHigh = status_vars[market_status]['rsiHigh']
            rsiLow = status_vars[market_status]['rsiLow']
            minBeta = status_vars[market_status]['minBeta']
            maxBeta = status_vars[market_status]['maxBeta']

            if (beta > minBeta and beta < maxBeta and smaLow < smaHigh and rsi < rsiHigh and rsi > rsiLow):
                log.debug(f"short: symbol: {symbol} | beta: {beta}")

                # if dollar volume change is in downward direction - more insiders are selling which means downward price indicator
                log.debug("short: getting insider data")
                end_date = (datetime.today().date() + timedelta(days=1))
                get_insider_transactions(symbol=symbol, start_date=config.start_date, end_date=end_date)
                _, df_insider = get_insider_from_db(symbol=symbol, tradingFrame=config.tradingFrame) # left return value is raw data, right is extended daily data

                if df_insider is None:
                    log.info(f"short: {symbol} insider data not available - skipping")
                    continue

                # remove data after cutoff date
                df_insider = df_insider[df_insider.index <= cutoff_date]
                log.info(f"short: insider df new index: {df_insider.index}")

                dollar_volume_series = df_insider['dollar_volume_change_insider']
                dollar_volume_30 = ta.ema(dollar_volume_series, length=30)
                dollar_volume_90 = ta.ema(dollar_volume_series, length=90)

                if dollar_volume_30 is None or dollar_volume_90 is None:
                    log.info(f"short: {symbol} dollar volume change data not available - skipping")
                    continue

                dollar_volume_30 = float(dollar_volume_30.iloc[-1])
                dollar_volume_90 = float(dollar_volume_90.iloc[-1])
                log.debug(f"short: {symbol} dollar volume ema 30: {dollar_volume_30} | ema 90: {dollar_volume_90}")

                # if insider transactions show that insiders are selling more stock - suggests negative trend
                if dollar_volume_30 < 0 and dollar_volume_30 < (0.9 * dollar_volume_90):
                    # next, get q earnings - save if downward trend
                    log.debug("short: getting earnings")
                    get_earnings(symbol=symbol, start_date=config.start_date)
                    df_earnings, _ = get_earnings_from_db(symbol=symbol, tradingFrame=config.tradingFrame, withTranscripts=False)

                    # no longer need earnings data - delete
                    QEarnings.objects.filter(symbol=symbol).delete()

                    if df_earnings is None:
                        log.info(f"short: {symbol} earnings data not available - skipping")
                        continue

                    # get latest earnings before cutoff date
                    df_earnings = df_earnings[df_earnings.index <= cutoff_date]
                    log.info(f"short: earnings new index: {df_earnings.index}")

                    # get market cap by price * shares outstanding
                    # get stock price the day of earnings report
                    url = 'https://data.alpaca.markets'
                    api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

                    start_date = df_earnings.index.max() - timedelta(days=5)
                    end_date = df_earnings.index.max()

                    try:
                        response = api.get_bars(symbol=symbol, timeframe='1Day', feed='sip', sort='asc', start=start_date.replace(tzinfo=None).date(), end=end_date.replace(tzinfo=None).date())
                    except Exception as err:
                        log.error(f"short: unable to get bars for {symbol} - skipping: {err}", exc_info=True)
                        continue 

                    stock_price = response[-1].c
                    shares_outstanding = df_earnings['shares_outstanding_qearnings'][-1]
                    market_cap = stock_price * shares_outstanding

                    if market_cap*1000000 < minCap:
                        log.info(f"short: market cap too low for {symbol} - {market_cap} - skipping stock")
                        continue

                    # determine if stock has downward financial trend based on scoring system 
                    # compare latest data to rolling averages of last 4 quarters
                    earnings_score = 0

                    indicators = ['eps', 'net_income', 'revenue', 'gross_margin', 'ebit', 'free_cash_flow', 'roe']

                    for indicator in indicators:
                        key = f'{indicator}_qearnings'
                        if key in df_earnings:
                            value = df_earnings[key]
                            log.debug(f"short: {symbol} - {indicator} value: {value}")
                            if value is None:
                                log.warning(f"short: {symbol} - {indicator} is None")
                                continue

                            value_sma = ta.sma(value, length=4)
                            if value_sma is None:
                                log.warning(f"short: {symbol} - {indicator} sma is None")
                                continue
                            
                            value_sma = value_sma.iloc[-1]
                            value = value.iloc[-1]
                            log.debug(f"short: {symbol} - {indicator} value sma: {value_sma}")
                            if value is None or value_sma is None:
                                log.warning(f"short: {symbol} - {indicator} last entry is None")
                                continue

                            value_sma = float(value_sma)
                            value = float(value)

                            log.debug(f"short: {indicator} | value: {value} | sma: {value_sma}")
                            if value < value_sma:
                                earnings_score += 1
                        else:
                            log.warning(f"{key} not in df")
                    
                    log.debug(f"short: final earnings score for {symbol}: {earnings_score}")

                    if earnings_score > 3:
                        log.debug(f"adding short symbol: {symbol}")
                        symbols_dict['symbol'].append(symbol)
                        symbols_dict['beta'].append(beta)
                        symbols_dict['market_cap'].append(market_cap)
                        # symbols_dict['eps'].append(eps)
                        # symbols_dict['roe'].append(roe)
            
            log.info(f"time to analyze {symbol}: {start_time}")
            start_time = time.time()

        log.debug(f"number of filtered short symbols: {len(symbols_dict['symbol'])}")

        # save symbols to db
        df = pd.DataFrame(symbols_dict)
        for i in range(len(df['symbol'])):
            ensure_connection() # ensure mysql connection
            ShortStocks(
                symbol=df['symbol'][i],
                beta=df['beta'][i],
                market_cap=df['market_cap'][i],
                # eps=df['eps'][i],
                # roe=df['roe'][i],
            ).save()
        return
    except Exception as err:
        log.error(f"error getting short symbols: {err}", exc_info=True)
        return None

'''
@param cutoff_date - datetime - current datetime
'''
def choose_long_symbols_new(symbols, market_status, minCap, cutoff_date):
    try:
        '''
        process: 
            - for each symbol
                - skip symbols we dont want - ie too low beta or market cap
                - get indicators
                - determine if price increased by x over the next 10 trading days
                - create random forests model for the symbol
                - predict if price will increase
                - save symbol if yes
        '''
        for symbol in symbols:
            log.info(f"analyzing {symbol}")

            '''
            indicators:
                - price
                - performance relative to SPY
                - volume 
                - atr
                - rsi
                - beta
                - gain streak
                - loss streak
                - num of closes greater than sma50 in the last 10 days
                - earnings
                - insider transactions
                - market cap
                - for most indictors above, get multiple rolling averages, crossovers, and divergence
            '''

            symbol_prices = {
                'date': [],
                'close': [],
                'high': [],
                'low': [],
                'volume': [],
            }
            url = 'https://data.alpaca.markets'
            api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

            # TODO: skip stocks that dont have price data before a specific date - lack of data
            response = api.get_bars(symbol=symbol, start=config.start_date, end=cutoff_date.replace(tzinfo=None).date(), timeframe='1Day')
            log.debug(f"{symbol} data first index: {response[0].t} | final index: {response[-1].t} | cutoff date: {cutoff_date}")
            for result in response:
                symbol_prices['date'].append(result.t.to_pydatetime())
                symbol_prices['close'].append(float(result.c))
                symbol_prices['high'].append(float(result.h))
                symbol_prices['low'].append(float(result.l))
                symbol_prices['volume'].append(float(result.v))

            symbol_df = pd.DataFrame(data=symbol_prices)
            symbol_df.set_index('date', inplace=True)
            symbol_df.index = symbol_df.index.tz_convert('US/Eastern')

            # price indicators
            symbol_df['symbol_sma50'] = ta.sma(symbol_df['close'], length=50)
            symbol_df['symbol_sma200'] = ta.sma(symbol_df['close'], length=200)

            symbol_df['symbol_sma50_diff'] = symbol_df['close'] - symbol_df['symbol_sma50']
            symbol_df['symbol_sma200_diff'] = symbol_df['close'] - symbol_df['symbol_sma200']

            prev_price = symbol_df['close'].shift(1)
            prev_sma50 = symbol_df['symbol_sma50'].shift(1)
            prev_sma200 = symbol_df['symbol_sma200'].shift(1)

            symbol_df['symbol_sma50_cross_above'] = ((symbol_df['close'] > symbol_df['symbol_sma50']) & (prev_price <= prev_sma50)).astype(int)
            symbol_df['symbol_sma200_cross_above'] = ((symbol_df['close'] > symbol_df['symbol_sma200']) & (prev_price <= prev_sma200)).astype(int)
            symbol_df['symbol_sma50_cross_below'] = ((symbol_df['close'] < symbol_df['symbol_sma50']) & (prev_price >= prev_sma50)).astype(int)
            symbol_df['symbol_sma200_cross_below'] = ((symbol_df['close'] < symbol_df['symbol_sma200']) & (prev_price >= prev_sma200)).astype(int)

            symbol_df['std50'] = ta.stdev(symbol_df['close'], length=50)
            symbol_df['std200'] = ta.stdev(symbol_df['close'], length=200)

            symbol_df['symbol_sma50_dvg_above'] = (symbol_df['close'] > (symbol_df['symbol_sma50'] + symbol_df['std50'])).astype(int)
            symbol_df['symbol_sma200_dvg_above'] = (symbol_df['close'] > (symbol_df['symbol_sma200'] + symbol_df['std200'])).astype(int)
            symbol_df['symbol_sma50_dvg_below'] = (symbol_df['close'] < (symbol_df['symbol_sma50'] - symbol_df['std50'])).astype(int)
            symbol_df['symbol_sma200_dvg_below'] = (symbol_df['close'] < (symbol_df['symbol_sma200'] - symbol_df['std200'])).astype(int)

            symbol_df.drop(columns=['std50', 'std200'], inplace=True)

            # atr data
            symbol_df['atr'] = ta.atr(high=symbol_df['high'], low=symbol_df['low'], close=symbol_df['close'], length=14)
            symbol_df['atr_sma50'] = ta.sma(symbol_df['atr'], length=50)

            symbol_df['atr_sma50_diff'] = symbol_df['atr'] - symbol_df['atr_sma50']

            prev_atr = symbol_df['atr'].shift(1)
            prev_atr_sma50 = symbol_df['atr_sma50'].shift(1)

            symbol_df['atr_sma50_cross_above'] = ((symbol_df['atr'] > symbol_df['atr_sma50']) & (prev_atr <= prev_atr_sma50)).astype(int)
            symbol_df['atr_sma50_cross_below'] = ((symbol_df['atr'] < symbol_df['atr_sma50']) & (prev_atr >= prev_atr_sma50)).astype(int)

            symbol_df['std50'] = ta.stdev(symbol_df['atr'], length=50)

            symbol_df['atr_sma50_dvg_above'] = (symbol_df['atr'] > (symbol_df['atr_sma50'] + symbol_df['std50'])).astype(int)
            symbol_df['atr_sma50_dvg_below'] = (symbol_df['atr'] < (symbol_df['atr_sma50'] - symbol_df['std50'])).astype(int)

            symbol_df.drop(columns=['high', 'low', 'volume', 'std50'], inplace=True) # no longer needed

            # beta 
            spy_prices = {
                'date': [],
                'close_spy': [],
            }
            response = api.get_bars(symbol='SPY', start=config.start_date, end=cutoff_date.replace(tzinfo=None).date(), timeframe='1Day')
            for result in response:
                spy_prices['date'].append(result.t.to_pydatetime())
                spy_prices['close_spy'].append(float(result.c))
            spy_df = pd.DataFrame(data=spy_prices)
            spy_df.set_index('date', inplace=True)
            spy_df.index = spy_df.index.tz_convert('US/Eastern')

            if len(spy_df['close_spy']) != len(symbol_df['close']):
                log.debug(f"not enough price data available for {symbol} - skipping")
                return
            spy_df.join(symbol_df[['close']].add_suffix('_symbol'), how='inner')

            spy_df = spy_df.pct_change().add_suffix('_return')
            rolling_cov = spy_df['close_symbol_return'].rolling(60).cov(spy_df['close_spy_return'])
            rolling_var = spy_df['spy_return'].rolling(60).var()
            symbol_df['symbol_beta'] = rolling_cov / rolling_var




            
        return 
    except Exception as err:
        log.error(f"error getting long stock symbols: {err}", exc_info=True)
        return

'''
@param symbols - list
@param market_status - str
@param minCap - int
@param cutoff_date - datetime
'''
def choose_long_symbols(symbols, market_status, minCap, cutoff_date):
    try:
        log.info("choosing long stocks")

        # variables depending on market status
        status_vars = {
            'bullish': {
                'rsiLow': 50,
                'rsiHigh': 999, # no limit
                'minBeta': 1.2, 
                'maxBeta': 2.5,
            }, 
            'neutral': {
                'rsiLow': 40,
                'rsiHigh': 60,
                'minBeta': 0.8,
                'maxBeta': 1.3,
            },
            'bearish': {
                'rsiLow': 25,
                'rsiHigh': 40,
                'minBeta': 0.0,
                'maxBeta': 0.8,
            }
        }

        # dictionary used to save symbols
        symbols_dict = {
            'symbol': [],
            'beta': [],
            'market_cap': [],
        }

        '''
        for each stock:
            - get sma, rsi, and beta
            - if all above threshold, get insider data
            - if rolling dollar volume is pointing upward, get quarterly earnings
            - if quarterly earnings are profitable and greater than rolling average, and market cap above threshold, save symbol
        '''
        for symbol in symbols:
            log.info(f"analyzing {symbol}")
            # slow down api calls per minute
            time.sleep(0.3)

            # make sure stock is tradable first
            url = config.alpaca.tradeUrl
            api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

            # put api call in try/catch incase it is not available
            try:
                asset = api.get_asset(symbol=symbol)
            except Exception:
                log.warning(f"asset not available - skipping: {symbol}")
                continue

            if not asset.tradable:
                log.info(f"{symbol} not tradable - skipping")
                continue
                
            beta = compute_beta(symbol=symbol, end_date=cutoff_date)

            # if sma10 less than sma50, save. otherwise, skip
            smaLow = get_sma(symbol=symbol, days=10, end_date=cutoff_date.replace(tzinfo=None).date())
            smaHigh = get_sma(symbol=symbol, days=50, end_date=cutoff_date.replace(tzinfo=None).date())

            # rsi based on market status
            rsi = get_rsi(symbol=symbol, days=14, end_date=cutoff_date.replace(tzinfo=None).date())

            log.debug(f"long: {symbol} smaLow: {smaLow} | smaHigh: {smaHigh} | rsi: {rsi} | beta {beta}")

            # some stocks missing data - skip
            if not beta or not smaLow or not smaHigh or not rsi:
                continue

            # beta greater than 1, market cap greater than 10 billion 10000000000, sma10 > sma50, and 55 <= rsi <= 70
            rsiHigh = status_vars[market_status]['rsiHigh']
            rsiLow = status_vars[market_status]['rsiLow']
            minBeta = status_vars[market_status]['minBeta']
            maxBeta = status_vars[market_status]['maxBeta']

            if (beta > minBeta and beta < maxBeta and smaLow > smaHigh and rsi < rsiHigh and rsi > rsiLow):
                log.debug(f"long: symbol: {symbol} | beta: {beta}")

                # if dollar volume change is in upward direction - more insiders are buying which means upward price indicator
                log.debug("long: getting insider data")
                end_date = datetime.today().date() + timedelta(days=1)
                get_insider_transactions(symbol=symbol, start_date=config.start_date, end_date=end_date)
                _, df_insider = get_insider_from_db(symbol=symbol, tradingFrame=config.tradingFrame) # left return value is raw data, right is extended daily data

                if df_insider is None:
                    log.info(f"long: {symbol} insider data not available - skipping")
                    continue

                # remove data after cutoff date
                df_insider = df_insider[df_insider.index <= cutoff_date]
                log.info(f"long: insider df new index: {df_insider.index}")

                dollar_volume_series = df_insider['dollar_volume_change_insider']
                dollar_volume_30 = ta.ema(dollar_volume_series, length=30)
                dollar_volume_90 = ta.ema(dollar_volume_series, length=90)

                if dollar_volume_30 is None or dollar_volume_90 is None:
                    log.info(f"long: {symbol} dollar volume change data not available - skipping")
                    continue

                dollar_volume_30 = float(dollar_volume_30.iloc[-1])
                dollar_volume_90 = float(dollar_volume_90.iloc[-1])
                log.debug(f"long: {symbol} dollar volume ema 30: {dollar_volume_30} | ema 90: {dollar_volume_90}")

                # if insider transactions show that insiders are buying more stock - suggests positive trend
                if dollar_volume_30 > 0 and dollar_volume_30 > (0.9 * dollar_volume_90):
                    # next, get q earnings and insider data - save if upward trending
                    log.debug("long: getting earnings")
                    get_earnings(symbol=symbol, start_date=config.start_date)
                    df_earnings, _ = get_earnings_from_db(symbol=symbol, tradingFrame=config.tradingFrame, withTranscripts=False)

                    # no longer need earnings data - delete
                    QEarnings.objects.filter(symbol=symbol).delete()

                    if df_earnings is None:
                        log.info(f"long: {symbol} earnings data not available - skipping")
                        continue

                    # get latest earnings before cutoff date
                    df_earnings = df_earnings[df_earnings.index <= cutoff_date]
                    log.info(f"long: earnings new index: {df_earnings.index}")

                    # get market cap by price * shares outstanding
                    # get stock price the day of earnings report
                    url = 'https://data.alpaca.markets'
                    api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

                    start_date = df_earnings.index.max() - timedelta(days=5)
                    end_date = df_earnings.index.max()

                    try:
                        response = api.get_bars(symbol=symbol, timeframe='1Day', feed='sip', sort='asc', start=start_date.replace(tzinfo=None).date(), end=end_date.replace(tzinfo=None).date())
                    except Exception as err:
                        log.error(f"long: unable to get bars for {symbol} - skipping: {err}", exc_info=True)
                        continue 

                    stock_price = response[-1].c
                    shares_outstanding = df_earnings['shares_outstanding_qearnings'][-1]
                    market_cap = stock_price * shares_outstanding

                    if market_cap*1000000 < minCap:
                        log.info(f"long: market cap too low for {symbol} - {market_cap} - skipping stock")
                        continue

                    # determine if stock has upward financial trend based on scoring system 
                    # compare latest data to rolling averages of last 4 quarters
                    earnings_score = 0

                    indicators = ['eps', 'net_income', 'revenue', 'gross_margin', 'ebit', 'free_cash_flow', 'roe']

                    for indicator in indicators:
                        key = f'{indicator}_qearnings'
                        if key in df_earnings:
                            value = df_earnings[key]
                            log.debug(f"long: {symbol} - {indicator} value: {value}")
                            if value is None:
                                log.warning(f"long: {symbol} - {indicator} is None")
                                continue

                            value_sma = ta.sma(value, length=4)
                            if value_sma is None:
                                log.warning(f"long: {symbol} - {indicator} sma is None")
                                continue
                            
                            value_sma = value_sma.iloc[-1]
                            value = value.iloc[-1]
                            log.debug(f"long: {symbol} - {indicator} value sma: {value_sma}")
                            if value is None or value_sma is None:
                                log.warning(f"long: {symbol} - {indicator} last entry is None")
                                continue

                            value_sma = float(value_sma)
                            value = float(value)

                            log.debug(f"long: {indicator} | value: {value} | sma: {value_sma}")
                            if value > value_sma:
                                earnings_score += 1
                        else:
                            log.warning(f"long: {key} not in df")
                    
                    log.debug(f"long: final earnings score for {symbol}: {earnings_score}")

                    if earnings_score > 3:
                        log.debug(f"adding long symbol: {symbol}")
                        symbols_dict['symbol'].append(symbol)
                        symbols_dict['beta'].append(beta)
                        symbols_dict['market_cap'].append(market_cap)
                        # symbols_dict['eps'].append(eps)
                        # symbols_dict['roe'].append(roe)

        log.debug(f"number of filtered long symbols: {len(symbols_dict['symbol'])}")

        # save symbols to db
        df = pd.DataFrame(symbols_dict)
        for i in range(len(df['symbol'])):
            ensure_connection() # ensure mysql connection
            LongStocks(
                symbol=df['symbol'][i],
                beta=df['beta'][i],
                market_cap=df['market_cap'][i],
                # eps=df['eps'][i],
                # roe=df['roe'][i],
            ).save()
        return 
    except Exception as err:
        log.error(f"error getting long stock symbols: {err}", exc_info=True)
        return
    
def choose_stocks(numStocks):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)

        # get long stocks
        long_query = "SELECT * FROM core_longstocks;"
        df_long = pd.read_sql(long_query, conn)

        # get top 'numStocks' based on max beta
        df_long = df_long.sort_values(by='beta', ascending=False, inplace=False)
        df_long = df_long.head(numStocks)

        long_symbols = list(df_long['symbol'])

        # get short symbols 
        short_query = "SELECT * FROM core_shortstocks;"
        df_short = pd.read_sql(short_query, conn)
        conn.close()

        # get top 'numStocks' based on min beta
        df_short = df_short.sort_values(by='beta', ascending=True, inplace=False)
        df_short = df_short.head(numStocks)

        short_symbols = list(df_short['symbol'])
        
        # remove duplicate entries - contradiction - shouldnt happen but just incase
        common_stocks = set(long_symbols) & set(short_symbols)

        long_symbols = [symbol for symbol in long_symbols if symbol not in common_stocks]
        short_symbols = [symbol for symbol in short_symbols if symbol not in common_stocks]
        return long_symbols, short_symbols
    except Exception as err:
        log.error(f"error getting stock symbols from db: {err}", exc_info=True)
        return None, None
    
'''
@param symbol - stock symbol
@param days - number of days for sma calculation
@param end_date - final date for stock price - date type
'''
def get_sma(symbol, days, end_date):
    try:
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        prices = []

        # get prices from the last x days - add extra to account for weekends - doesnt work if we set day to too high
        start_date = end_date - timedelta(days=400)
        response = api.get_bars(symbol=symbol, start=start_date, end=end_date, timeframe='1Day')
        time.sleep(0.3)

        # add prices to list
        for result in response:
            prices.append(float(result.c))

        # convert data to pandas series before getting calculations
        prices = pd.Series(prices)

        sma = ta.sma(prices, length=days)

        if sma is None:
            log.warning(f"sma is none for {symbol} - skipping")
            return None

        # get latest value from series and convert to float
        sma = float(sma.iloc[-1])

        return sma
    except Exception as error:
        log.error(f"error getting sma for {symbol}", exc_info=True)
        return None
    
'''
@param symbol - stock symbol
@param days - number of days for sma calculation
@param end_date - final date for stock price - date type
'''
def get_rsi(symbol, days, end_date):
    try:
        # edge case for bullish market
        if days is None:
            return 9999999

        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        prices = []

        # get prices from the last x days - add extra to account for weekends - doesnt work if we set day to too high
        start_date = end_date - timedelta(days=400)
        response = api.get_bars(symbol=symbol, start=start_date, end=end_date, timeframe='1Day')
        time.sleep(0.3)

        # add prices to list
        for result in response:
            prices.append(float(result.c))

        # convert data to pandas series before getting calculations
        prices = pd.Series(prices)

        rsi = ta.rsi(prices, length=days)

        if rsi is None:
            log.warning(f"rsi is none for {symbol} - skipping")
            return None

        # get latest value from series and convert to float
        rsi = float(rsi.iloc[-1])

        return rsi
    except Exception as error:
        log.error(f"error getting rsi for {symbol}", exc_info=True)
        return None

'''
@param cutoff_date - datetime
'''
def get_short_term_market_status(cutoff_date):
    try:
        log.debug(f"getting short-term market status")
        get_ad_ratio(retry_attempts=3, end_date=datetime.now(tz=pytz.timezone("US/Eastern")), timeframe='weekly')

        # get spy data and indicators 
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        spy_prices = {
            'date': [],
            'close': [],
            'high': [],
            'low': [],
        }
        response = api.get_bars(symbol='SPY', start=config.start_date, end=cutoff_date.replace(tzinfo=None).date(), timeframe='1Day', limit=10000)
        log.debug(f"SPY data first index: {response[0].t} | final index: {response[-1].t} | cutoff date: {cutoff_date}")
        for result in response:
            spy_prices['date'].append(result.t.to_pydatetime())
            spy_prices['close'].append(float(result.c))
            spy_prices['high'].append(float(result.h))
            spy_prices['low'].append(float(result.l))
        
        spy_df = pd.DataFrame(data=spy_prices)
        spy_df.set_index('date', inplace=True)
        spy_df.index = spy_df.index.tz_convert('US/Eastern')

        # price data
        spy_df['spy_sma50'] = ta.sma(spy_df['close'], length=50)
        spy_df['spy_sma200'] = ta.sma(spy_df['close'], length=200)

        spy_df['spy_sma50_diff'] = spy_df['close'] - spy_df['spy_sma50']
        spy_df['spy_sma200_diff'] = spy_df['close'] - spy_df['spy_sma200']

        prev_price = spy_df['close'].shift(1)
        prev_sma50 = spy_df['spy_sma50'].shift(1)
        prev_sma200 = spy_df['spy_sma200'].shift(1)

        spy_df['spy_sma50_cross_above'] = ((spy_df['close'] > spy_df['spy_sma50']) & (prev_price <= prev_sma50)).astype(int)
        spy_df['spy_sma200_cross_above'] = ((spy_df['close'] > spy_df['spy_sma200']) & (prev_price <= prev_sma200)).astype(int)
        spy_df['spy_sma50_cross_below'] = ((spy_df['close'] < spy_df['spy_sma50']) & (prev_price >= prev_sma50)).astype(int)
        spy_df['spy_sma200_cross_below'] = ((spy_df['close'] < spy_df['spy_sma200']) & (prev_price >= prev_sma200)).astype(int)

        spy_df['std50'] = ta.stdev(spy_df['close'], length=50)
        spy_df['std200'] = ta.stdev(spy_df['close'], length=200)

        spy_df['spy_sma50_dvg_above'] = (spy_df['close'] > (spy_df['spy_sma50'] + spy_df['std50'])).astype(int)
        spy_df['spy_sma200_dvg_above'] = (spy_df['close'] > (spy_df['spy_sma200'] + spy_df['std200'])).astype(int)
        spy_df['spy_sma50_dvg_below'] = (spy_df['close'] < (spy_df['spy_sma50'] - spy_df['std50'])).astype(int)
        spy_df['spy_sma200_dvg_below'] = (spy_df['close'] < (spy_df['spy_sma200'] - spy_df['std200'])).astype(int)

        spy_df.drop(columns=['std50', 'std200'], inplace=True)

        # atr data
        spy_df['atr'] = ta.atr(high=spy_df['high'], low=spy_df['low'], close=spy_df['close'], length=14)

        atr_sma10 = ta.sma(spy_df['atr'], length=10)
        spy_df['atr_spike'] = (spy_df['atr'] > (1.1 * atr_sma10)).astype(int)
        spy_df['atr_spike_smoothed'] = spy_df['atr'] / atr_sma10
        spy_df['atr_sma50'] = ta.sma(spy_df['atr'], length=50)

        spy_df['atr_sma50_diff'] = spy_df['atr'] - spy_df['atr_sma50']

        prev_atr = spy_df['atr'].shift(1)
        prev_atr_sma50 = spy_df['atr_sma50'].shift(1)

        spy_df['atr_sma50_cross_above'] = ((spy_df['atr'] > spy_df['atr_sma50']) & (prev_atr <= prev_atr_sma50)).astype(int)
        spy_df['atr_sma50_cross_below'] = ((spy_df['atr'] < spy_df['atr_sma50']) & (prev_atr >= prev_atr_sma50)).astype(int)

        spy_df['std50'] = ta.stdev(spy_df['atr'], length=50)

        spy_df['atr_sma50_dvg_above'] = (spy_df['atr'] > (spy_df['atr_sma50'] + spy_df['std50'])).astype(int)
        spy_df['atr_sma50_dvg_below'] = (spy_df['atr'] < (spy_df['atr_sma50'] - spy_df['std50'])).astype(int)

        spy_df.drop(columns=['high', 'low', 'std50'], inplace=True) # no longer needed

        spy_df = spy_df[spy_df.index.weekday == 4]
        log.debug(f"spy df new index: {spy_df.index} | length: {len(spy_df.index)}")

        # get ad ratio for each week - returns a df - outer join with spy_df and fill missing entries with 0
        ad_ratio_df = get_ad_ratio(retry_attempts=3, end_date=cutoff_date, timeframe='weekly')
        ad_ratio_df.index = ad_ratio_df.index.tz_convert('US/Eastern')
        log.debug(f"ad ratio data: {ad_ratio_df['ad_ratio']}")

        spy_df = spy_df.join(ad_ratio_df, how='outer').fillna(0)
        log.debug(f"merged ad ratio data: {ad_ratio_df['ad_ratio']}")
        spy_df['ad_ratio_diff'] = spy_df['ad_ratio'].diff() 
        
        # do sma10 for ad ratio since we have weekly data now
        spy_df['ad_ratio_sma10'] = ta.sma(spy_df['ad_ratio'], length=10)
        spy_df['ad_ratio_sma10_diff'] = spy_df['ad_ratio'] - spy_df['ad_ratio_sma10']

        prev_ad_ratio = spy_df['ad_ratio'].shift(1)
        prev_ad_ratio_sma10 = spy_df['ad_ratio_sma10'].shift(1)

        spy_df['ad_ratio_sma10_cross_above'] = ((spy_df['ad_ratio'] > spy_df['ad_ratio_sma10']) & (prev_ad_ratio <= prev_ad_ratio_sma10)).astype(int)
        spy_df['ad_ratio_sma10_cross_below'] = ((spy_df['ad_ratio'] < spy_df['ad_ratio_sma10']) & (prev_ad_ratio >= prev_ad_ratio_sma10)).astype(int)

        spy_df['std10'] = ta.stdev(spy_df['ad_ratio'], length=10)
        spy_df['ad_ratio_sma10_dvg_above'] = (spy_df['ad_ratio'] > (spy_df['ad_ratio_sma10'] + spy_df['std10'])).astype(int)
        spy_df['ad_ratio_sma10_dvg_below'] = (spy_df['ad_ratio'] < (spy_df['ad_ratio_sma10'] - spy_df['std10'])).astype(int)
        spy_df.drop(columns=['std10'], inplace=True)
        
        # get y-vars - if close increased 10 trading days later = 1, 0 otherwise
        pos_pct_change_5d = [0] * len(spy_df['close'])
        total_bulls = 0
        neg_pct_change_5d = [0] * len(spy_df['close'])
        total_bears = 0
        for i in range(len(spy_df['close']) - 1):
            pct_change = (spy_df['close'].iloc[i + 1] - spy_df['close'].iloc[i]) / np.abs(spy_df['close'].iloc[i])
            if pct_change > config.stocks.short_term_market.pos_pct_change:
                pos_pct_change_5d[i] = 1
                total_bulls += 1

            if pct_change < config.stocks.short_term_market.neg_pct_change:
                neg_pct_change_5d[i] = 1
                total_bears += 1 
        log.debug(f"{cutoff_date} short term total training bulls: {total_bulls} | total training bears: {total_bears}")
        spy_df['pos_pct_change_5d'] = pd.Series(pos_pct_change_5d, index=spy_df.index)
        spy_df['neg_pct_change_5d'] = pd.Series(neg_pct_change_5d, index=spy_df.index)
        # spy_df.drop(columns=['close'], inplace=True) # no longer needed

        log.debug(f"{cutoff_date} short term number of training bearish responses: {sum(spy_df['neg_pct_change_5d'])} | number of training atr spikes: {sum(spy_df['atr_spike'])}")
        for i in range(len(spy_df.index)):
            if spy_df['neg_pct_change_5d'][i] == 1 and spy_df['atr_spike'][i] == 1:
                log.debug(f"both bearish response and atr spike are 1 on {spy_df.index[i]}")

        # get market news 
        df_market_news = get_market_news_from_db(tradingFrame='weekly')
        df_market_news = df_market_news[df_market_news.index <= cutoff_date]

        # merge data 
        log.debug(f"weekly spy index: {spy_df.index}")
        log.debug(f"weekly news index: {df_market_news.index}")
        df = spy_df.join(df_market_news, how='inner')
        log.debug(f"joined df index: {df.index}")
        df_truncated = df.copy()

        # dont want latest date to be in trained model - we are predicting this
        df_truncated = df_truncated[df.index < cutoff_date]
        log.debug(f"df_truncated last index for {cutoff_date} prediction: {df_truncated.index[-1]}")

        # train model
        features = df_truncated.columns.tolist()
        features.remove('pos_pct_change_5d')
        features.remove('neg_pct_change_5d')
        log.debug(f"features: {features}")

        # get best feature combination for each 
        num_features = len(features)
        log.debug(f"number of total features for short term models: {num_features}")

        # create bullish model
        bull_features, bull_prec, bull_acc = get_best_feature_combination(df=df_truncated.drop(columns=['neg_pct_change_5d']), signal_col_name='pos_pct_change_5d', training_fxn=seq_train_model, balanced=False, num_combs=1000, min_comb_len=int(0.2 * num_features), max_comb_len=int(0.45 * num_features))
        log.debug(f"{cutoff_date} short term best bull precision: {bull_prec} | acc: {bull_acc} | features: {bull_features}")
        if bull_features != []: # save model if successful features found
            _, pos_acc, pos_prec, _ = seq_train_model(df=df_truncated, features=bull_features, target_y='pos_pct_change_5d', balanced=False, fileName="symbolSelection/shortTermBullishMarketStatus")
            log.debug(f"short term precision, accuracy and features for predicting bullish market status using sequential sampling: {pos_prec} | {pos_acc} | {bull_features}")
        else: # if bull model failed, revert to random training
            log.warning(f"short term no strong model returned for bullish - reverting to random training")
            bull_features, bull_prec, bull_acc = get_best_feature_combination(df=df_truncated.drop(columns=['neg_pct_change_5d']), signal_col_name='pos_pct_change_5d', training_fxn=train_model, balanced=False, num_combs=1000, min_comb_len=int(0.2 * num_features), max_comb_len=int(0.45 * num_features))
            log.debug(f"{cutoff_date} short term best bull precision using random sampling: {bull_prec} | acc: {bull_acc} | features: {bull_features}")
            if bull_features == []:
                log.warning(f"short term no strong model returned for bullish - reverting to all features")
                bull_features = features 

            _, pos_acc, pos_prec, _ = train_model(df=df_truncated, features=bull_features, target_y='pos_pct_change_5d', balanced=False, fileName='symbolSelection/shortTermBullishMarketStatus')
            log.debug(f"short term precision, accuracy and features for predicting bullish market status using random sampling: {pos_prec} | {pos_acc} | {bull_features}")

        # create bearish model
        bear_features, bear_prec, bear_acc = get_best_feature_combination(df=df_truncated.drop(columns=['pos_pct_change_5d']), signal_col_name='neg_pct_change_5d', training_fxn=seq_train_model, balanced=False, num_combs=2000, min_comb_len=int(0.2 * num_features), max_comb_len=int(0.45 * num_features))
        log.debug(f"{cutoff_date} short term best bear precision: {bear_prec} | acc: {bear_acc} | features: {bear_features}")
        if bear_features != []: # save model if successful features found
            _, neg_acc, neg_prec, _ = seq_train_model(df=df_truncated, features=bear_features, target_y='neg_pct_change_5d', balanced=False, fileName='symbolSelection/shortTermBearishMarketStatus')
            log.debug(f"short term precision, accuracy and features for predicting bearish market status using sequential sampling: {neg_prec} | {neg_acc} | {bear_features}")
        else: # if bear model failed, revert to random training
            log.warning(f"short term no strong model returned for bearish - reverting to random training")
            bear_features, bear_prec, bear_acc = get_best_feature_combination(df=df_truncated.drop(columns=['pos_pct_change_5d']), signal_col_name='neg_pct_change_5d', training_fxn=train_model, balanced=False, num_combs=1000, min_comb_len=int(0.2 * num_features), max_comb_len=int(0.45 * num_features))
            log.debug(f"{cutoff_date} short term best bear precision using random sampling: {bear_prec} | acc: {bear_acc} | features: {bear_features}")
            if bear_features == []:
                log.warning(f"short term no strong model returned for bearish - reverting to all features")
                bear_features = features 

            _, neg_acc, neg_prec, _ = train_model(df=df_truncated, features=bear_features, target_y='neg_pct_change_5d', balanced=False, fileName='symbolSelection/shortTermBearishMarketStatus')
            log.debug(f"short term precision, accuracy and features for predicting bearish market status using random sampling: {neg_prec} | {neg_acc} | {bear_features}")

        # load latest data and predict next weeks market status
        # get features we need for this model
        df_bullish = df[bull_features]
        df_bearish = df[bear_features]
        
        # load ai model 
        bullish_model = joblib.load(f"models/symbolSelection/shortTermBullishMarketStatus.joblib")
        bearish_model = joblib.load(f"models/symbolSelection/shortTermBearishMarketStatus.joblib")

        # get most recent data to predict
        bullish_entry = df_bullish.iloc[-1]
        bearish_entry = df_bearish.iloc[-1]
        log.debug(f"short term test simulation {cutoff_date}: associated index with data: {df.index[-1]}")

        # model expect 2d array of entry
        bullish_entry_arr = bullish_entry.values.reshape(1, -1)
        bearish_entry_arr = bearish_entry.values.reshape(1, -1)

        # make prediction
        bull_prediction = int(bullish_model.predict(bullish_entry_arr))
        bear_prediction = int(bearish_model.predict(bearish_entry_arr))
        log.debug(f"{cutoff_date} short term bull prediction: {bull_prediction} | bear prediction: {bear_prediction}")

        # bearish models are inconsistent - if precision is bad, overwrite prediction to 0
        if neg_prec < config.models.minPrecision:
            log.info(f"short term bearish model precision too low: {neg_prec} - overwriting prediction")
            bear_prediction = 0

        log.debug(f"{cutoff_date} short term atr spike value: {df['atr_spike'][-1]} and date: {df.index[-1]}")
        if df['atr_spike'][-1] == 0:
            log.info(f"{cutoff_date} short term latest atr spike is 0 - defaulting bearish prediction to 0")
            bear_prediction = 0

        # for now, log feature importance on models that predicted 1 - want to see whats causing false positives
        if bull_prediction == 1:
            bull_importances = bullish_model.feature_importances_
            bull_feature_importance_df = pd.DataFrame({
                'features': bull_features,
                'importance': bull_importances,
            }).sort_values(by='importance', ascending=False)
            log.debug(f"short term bullish features importance: {bull_feature_importance_df}")

        if bear_prediction == 1:
            bear_importances = bearish_model.feature_importances_
            bear_feature_importance_df = pd.DataFrame({
                'features': bear_features,
                'importance': bear_importances,
            }).sort_values(by='importance', ascending=False)
            log.debug(f"short term bearish features importance: {bear_feature_importance_df}")

        final_prediction = 0
        if bull_prediction == 1 and bear_prediction == 0:
            final_prediction = 1
        if bull_prediction == 0 and bear_prediction == 1:
            final_prediction = -1
        
        return bull_prediction, bear_prediction, final_prediction
    except Exception as err:
        log.error(f"error getting short-term market status - defaulting to neutral to be safe: {err}", exc_info=True)

        return 0, 0, 0
'''
@param cutoff_date - datetime
'''
def get_med_term_market_status(cutoff_date):
    try:
        log.debug(f"getting med-term market status")
        get_ad_ratio(retry_attempts=3, end_date=datetime.now(tz=pytz.timezone("US/Eastern")), timeframe='weekly')
        get_macro(start_date=config.start_date)
        _, df_macro = get_macro_from_db(tradingFrame=config.tradingFrame)

        # get latest data before cutoff date
        df_macro = df_macro[df_macro.index <= cutoff_date]
        df_macro = df_macro[['ism_pmi_macro_exp', 'industrial_production_mom_macro_exp', 'cpi_macro_exp', 'unemployment_rate_macro_exp', 'consumer_sentiment_macro_exp', 'ism_pmi_sma4_macro_exp', 'industrial_production_mom_sma4_macro_exp', 'cpi_sma4_macro_exp', 'unemployment_rate_sma4_macro_exp', 'consumer_sentiment_sma4_macro_exp', 'ism_pmi_sma4_diff_macro_exp', 'industrial_production_mom_sma4_diff_macro_exp', 'cpi_sma4_diff_macro_exp', 'unemployment_rate_sma4_diff_macro_exp', 'consumer_sentiment_sma4_diff_macro_exp', 'ism_pmi_diff_macro_exp', 'industrial_production_mom_diff_macro_exp', 'cpi_diff_macro_exp', 'unemployment_rate_diff_macro_exp', 'consumer_sentiment_diff_macro_exp', 'ism_pmi_cross_above_macro_exp', 'industrial_production_mom_cross_above_macro_exp', 'cpi_cross_above_macro_exp', 'unemployment_rate_cross_above_macro_exp', 'consumer_sentiment_cross_above_macro_exp', 'ism_pmi_cross_below_macro_exp', 'industrial_production_mom_cross_below_macro_exp', 'cpi_cross_below_macro_exp', 'unemployment_rate_cross_below_macro_exp', 'consumer_sentiment_cross_below_macro_exp', 'ism_pmi_dvg_above_macro_exp', 'industrial_production_mom_dvg_above_macro_exp', 'cpi_dvg_above_macro_exp', 'unemployment_rate_dvg_above_macro_exp', 'consumer_sentiment_dvg_above_macro_exp', 'ism_pmi_dvg_below_macro_exp', 'industrial_production_mom_dvg_below_macro_exp', 'cpi_dvg_below_macro_exp', 'unemployment_rate_dvg_below_macro_exp', 'consumer_sentiment_dvg_below_macro_exp',]]

        log.info(f"cut off date: {cutoff_date} new macro data index: {df_macro.index}")
        log.debug(f"df macro: {df_macro['ism_pmi_macro_exp']}")

        # split into weekly data - if index is a friday
        df_macro_weekly = df_macro[df_macro.index.weekday == 4]
        log.debug(f"weekly macro df index: {df_macro_weekly.index}")

        # get spy data and indicators 
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        spy_prices = {
            'date': [],
            'close': [],
            'high': [],
            'low': [],
        }
        response = api.get_bars(symbol='SPY', start=config.start_date, end=cutoff_date.replace(tzinfo=None).date(), timeframe='1Day', limit=10000)
        log.debug(f"SPY data first index: {response[0].t} | final index: {response[-1].t} | cutoff date: {cutoff_date}")
        for result in response:
            spy_prices['date'].append(result.t.to_pydatetime())
            spy_prices['close'].append(float(result.c))
            spy_prices['high'].append(float(result.h))
            spy_prices['low'].append(float(result.l))
        
        spy_df = pd.DataFrame(data=spy_prices)
        spy_df.set_index('date', inplace=True)
        spy_df.index = spy_df.index.tz_convert('US/Eastern')

        # price data
        spy_df['spy_sma50'] = ta.sma(spy_df['close'], length=50)
        spy_df['spy_sma200'] = ta.sma(spy_df['close'], length=200)

        spy_df['spy_sma50_diff'] = spy_df['close'] - spy_df['spy_sma50']
        spy_df['spy_sma200_diff'] = spy_df['close'] - spy_df['spy_sma200']

        prev_price = spy_df['close'].shift(1)
        prev_sma50 = spy_df['spy_sma50'].shift(1)
        prev_sma200 = spy_df['spy_sma200'].shift(1)

        spy_df['spy_sma50_cross_above'] = ((spy_df['close'] > spy_df['spy_sma50']) & (prev_price <= prev_sma50)).astype(int)
        spy_df['spy_sma200_cross_above'] = ((spy_df['close'] > spy_df['spy_sma200']) & (prev_price <= prev_sma200)).astype(int)
        spy_df['spy_sma50_cross_below'] = ((spy_df['close'] < spy_df['spy_sma50']) & (prev_price >= prev_sma50)).astype(int)
        spy_df['spy_sma200_cross_below'] = ((spy_df['close'] < spy_df['spy_sma200']) & (prev_price >= prev_sma200)).astype(int)

        spy_df['std50'] = ta.stdev(spy_df['close'], length=50)
        spy_df['std200'] = ta.stdev(spy_df['close'], length=200)

        spy_df['spy_sma50_dvg_above'] = (spy_df['close'] > (spy_df['spy_sma50'] + spy_df['std50'])).astype(int)
        spy_df['spy_sma200_dvg_above'] = (spy_df['close'] > (spy_df['spy_sma200'] + spy_df['std200'])).astype(int)
        spy_df['spy_sma50_dvg_below'] = (spy_df['close'] < (spy_df['spy_sma50'] - spy_df['std50'])).astype(int)
        spy_df['spy_sma200_dvg_below'] = (spy_df['close'] < (spy_df['spy_sma200'] - spy_df['std200'])).astype(int)

        spy_df.drop(columns=['std50', 'std200'], inplace=True)

        # atr data
        spy_df['atr'] = ta.atr(high=spy_df['high'], low=spy_df['low'], close=spy_df['close'], length=14)

        atr_sma10 = ta.sma(spy_df['atr'], length=10)
        spy_df['atr_spike'] = (spy_df['atr'] > (1.1 * atr_sma10)).astype(int)
        spy_df['atr_spike_smoothed'] = spy_df['atr'] / atr_sma10
        spy_df['atr_sma50'] = ta.sma(spy_df['atr'], length=50)

        spy_df['atr_sma50_diff'] = spy_df['atr'] - spy_df['atr_sma50']

        prev_atr = spy_df['atr'].shift(1)
        prev_atr_sma50 = spy_df['atr_sma50'].shift(1)

        spy_df['atr_sma50_cross_above'] = ((spy_df['atr'] > spy_df['atr_sma50']) & (prev_atr <= prev_atr_sma50)).astype(int)
        spy_df['atr_sma50_cross_below'] = ((spy_df['atr'] < spy_df['atr_sma50']) & (prev_atr >= prev_atr_sma50)).astype(int)

        spy_df['std50'] = ta.stdev(spy_df['atr'], length=50)

        spy_df['atr_sma50_dvg_above'] = (spy_df['atr'] > (spy_df['atr_sma50'] + spy_df['std50'])).astype(int)
        spy_df['atr_sma50_dvg_below'] = (spy_df['atr'] < (spy_df['atr_sma50'] - spy_df['std50'])).astype(int)

        spy_df.drop(columns=['high', 'low', 'std50'], inplace=True) # no longer needed

        spy_df = spy_df[spy_df.index.weekday == 4]
        log.debug(f"spy df new index: {spy_df.index} | length: {len(spy_df.index)}")

        # get ad ratio for each week - returns a df - outer join with spy_df and fill missing entries with 0
        ad_ratio_df = get_ad_ratio(retry_attempts=3, end_date=cutoff_date, timeframe='weekly')
        ad_ratio_df.index = ad_ratio_df.index.tz_convert('US/Eastern')
        log.debug(f"ad ratio data: {ad_ratio_df['ad_ratio']}")

        spy_df = spy_df.join(ad_ratio_df, how='outer').fillna(0)
        log.debug(f"merged ad ratio data: {ad_ratio_df['ad_ratio']}")
        spy_df['ad_ratio_diff'] = spy_df['ad_ratio'].diff() 
        
        # do sma10 for ad ratio since we have weekly data now
        spy_df['ad_ratio_sma10'] = ta.sma(spy_df['ad_ratio'], length=10)
        spy_df['ad_ratio_sma10_diff'] = spy_df['ad_ratio'] - spy_df['ad_ratio_sma10']

        prev_ad_ratio = spy_df['ad_ratio'].shift(1)
        prev_ad_ratio_sma10 = spy_df['ad_ratio_sma10'].shift(1)

        spy_df['ad_ratio_sma10_cross_above'] = ((spy_df['ad_ratio'] > spy_df['ad_ratio_sma10']) & (prev_ad_ratio <= prev_ad_ratio_sma10)).astype(int)
        spy_df['ad_ratio_sma10_cross_below'] = ((spy_df['ad_ratio'] < spy_df['ad_ratio_sma10']) & (prev_ad_ratio >= prev_ad_ratio_sma10)).astype(int)

        spy_df['std10'] = ta.stdev(spy_df['ad_ratio'], length=10)
        spy_df['ad_ratio_sma10_dvg_above'] = (spy_df['ad_ratio'] > (spy_df['ad_ratio_sma10'] + spy_df['std10'])).astype(int)
        spy_df['ad_ratio_sma10_dvg_below'] = (spy_df['ad_ratio'] < (spy_df['ad_ratio_sma10'] - spy_df['std10'])).astype(int)
        spy_df.drop(columns=['std10'], inplace=True)

        # RASI indicators 
        net_advance = spy_df['advanced'] - spy_df['declined']
        net_advance_ema4 = ta.ema(net_advance, length=4)
        net_advance_ema9 = ta.ema(net_advance, length=9)
        breadth_mtm = net_advance_ema4 - net_advance_ema9
        spy_df['RASI'] = breadth_mtm.cumsum()

        spy_return = spy_df['close'].pct_change()
        rasi_slope = spy_df['RASI'].diff()
        spy_df['breadth_divergence'] = ((spy_return > 0) & (rasi_slope < 0)).astype(int)
        spy_df['breadth_divergence_magnitude'] = spy_return - rasi_slope
        spy_df['bullish_breakdown'] = ((spy_df['close'] > spy_df['spy_sma50']) & (spy_df['RASI'] < 0)).astype(int)
        
        # get y-vars - if close increased 10 trading days later = 1, 0 otherwise
        pos_pct_change_10d = [0] * len(spy_df['close'])
        total_bulls = 0
        neg_pct_change_10d = [0] * len(spy_df['close'])
        total_bears = 0
        for i in range(len(spy_df['close']) - 2):
            pct_change = (spy_df['close'].iloc[i + 2] - spy_df['close'].iloc[i]) / np.abs(spy_df['close'].iloc[i])
            if pct_change > config.stocks.med_term_market.pos_pct_change:
                pos_pct_change_10d[i] = 1
                total_bulls += 1

            if pct_change < config.stocks.med_term_market.neg_pct_change:
                neg_pct_change_10d[i] = 1
                total_bears += 1 
        log.debug(f"{cutoff_date} med term total training bulls: {total_bulls} | total training bears: {total_bears}")
        spy_df['pos_pct_change_10d'] = pd.Series(pos_pct_change_10d, index=spy_df.index)
        spy_df['neg_pct_change_10d'] = pd.Series(neg_pct_change_10d, index=spy_df.index)
        # spy_df.drop(columns=['close'], inplace=True) # no longer needed

        log.debug(f"{cutoff_date} med term number of training bearish responses: {sum(spy_df['neg_pct_change_10d'])} | number of training atr spikes: {sum(spy_df['atr_spike'])}")
        for i in range(len(spy_df.index)):
            if spy_df['neg_pct_change_10d'][i] == 1 and spy_df['atr_spike'][i] == 1:
                log.debug(f"med term both bearish response and atr spike are 1 on {spy_df.index[i]}")

        # get market news 
        df_market_news = get_market_news_from_db(tradingFrame='weekly')
        df_market_news = df_market_news[df_market_news.index <= cutoff_date]

        # merge data 
        log.debug(f"weekly macro index: {df_macro_weekly.index}")
        log.debug(f"weekly spy index: {spy_df.index}")
        log.debug(f"weekly news index: {df_market_news.index}")
        df = df_macro_weekly.join(spy_df, how='inner')
        df = df.join(df_market_news, how='inner')
        log.debug(f"joined df index: {df.index}")
        df.to_csv('macro.csv', index=True)
        df_truncated = df.copy()

        # dont want latest date to be in trained model - we are predicting this
        df_truncated = df_truncated[df.index < cutoff_date]
        log.debug(f"df_truncated last index for {cutoff_date} prediction: {df_truncated.index[-1]}")

        # train model
        features = df_truncated.columns.tolist()
        features.remove('pos_pct_change_10d')
        features.remove('neg_pct_change_10d')
        log.debug(f"features: {features}")

        # get best feature combination for each 
        num_features = len(features)
        log.debug(f"number of total features for med term models: {num_features}")

        # create bullish model
        bull_features, bull_prec, bull_acc = get_best_feature_combination(df=df_truncated.drop(columns=['neg_pct_change_10d']), signal_col_name='pos_pct_change_10d', training_fxn=seq_train_model, balanced=False, num_combs=1000, min_comb_len=int(0.1 * num_features), max_comb_len=int(0.35 * num_features))
        log.debug(f"{cutoff_date} med term best bull precision: {bull_prec} | acc: {bull_acc} | features: {bull_features}")
        if bull_features != []: # save model if successful features found
            _, pos_acc, pos_prec, _ = seq_train_model(df=df_truncated, features=bull_features, target_y='pos_pct_change_10d', balanced=False, fileName="symbolSelection/medTermBullishMarketStatus")
            log.debug(f"med term precision, accuracy and features for predicting bullish market status using sequential sampling: {pos_prec} | {pos_acc} | {bull_features}")
        else: # if bull model failed, revert to random training
            log.warning(f"med term no strong model returned for bullish - reverting to random training")
            bull_features, bull_prec, bull_acc = get_best_feature_combination(df=df_truncated.drop(columns=['neg_pct_change_10d']), signal_col_name='pos_pct_change_10d', training_fxn=train_model, balanced=False, num_combs=1000, min_comb_len=int(0.1 * num_features), max_comb_len=int(0.35 * num_features))
            log.debug(f"{cutoff_date} med term best bull precision using random sampling: {bull_prec} | acc: {bull_acc} | features: {bull_features}")
            if bull_features == []:
                log.warning(f"med term no strong model returned for bullish - reverting to all features")
                bull_features = features 

            _, pos_acc, pos_prec, _ = train_model(df=df_truncated, features=bull_features, target_y='pos_pct_change_10d', balanced=False, fileName='symbolSelection/medTermBullishMarketStatus')
            log.debug(f"med term precision, accuracy and features for predicting bullish market status using random sampling: {pos_prec} | {pos_acc} | {bull_features}")

        # create bearish model
        bear_features, bear_prec, bear_acc = get_best_feature_combination(df=df_truncated.drop(columns=['pos_pct_change_10d']), signal_col_name='neg_pct_change_10d', training_fxn=seq_train_model, balanced=False, num_combs=2000, min_comb_len=int(0.1 * num_features), max_comb_len=int(0.35 * num_features))
        log.debug(f"{cutoff_date} med term best bear precision: {bear_prec} | acc: {bear_acc} | features: {bear_features}")
        if bear_features != []: # save model if successful features found
            _, neg_acc, neg_prec, _ = seq_train_model(df=df_truncated, features=bear_features, target_y='neg_pct_change_10d', balanced=False, fileName='symbolSelection/medTermBearishMarketStatus')
            log.debug(f"med term precision, accuracy and features for predicting bearish market status using sequential sampling: {neg_prec} | {neg_acc} | {bear_features}")
        else: # if bear model failed, revert to random training
            log.warning(f"med term no strong model returned for bearish - reverting to random training")
            bear_features, bear_prec, bear_acc = get_best_feature_combination(df=df_truncated.drop(columns=['pos_pct_change_10d']), signal_col_name='neg_pct_change_10d', training_fxn=train_model, balanced=False, num_combs=1000, min_comb_len=int(0.1 * num_features), max_comb_len=int(0.35 * num_features))
            log.debug(f"{cutoff_date} med term best bear precision using random sampling: {bear_prec} | acc: {bear_acc} | features: {bear_features}")
            if bear_features == []:
                log.warning(f"med term no strong model returned for bearish - reverting to all features")
                bear_features = features 

            _, neg_acc, neg_prec, _ = train_model(df=df_truncated, features=bear_features, target_y='neg_pct_change_10d', balanced=False, fileName='symbolSelection/medTermBearishMarketStatus')
            log.debug(f"med term precision, accuracy and features for predicting bearish market status using random sampling: {neg_prec} | {neg_acc} | {bear_features}")

        # load latest data and predict next weeks market status
        # get features we need for this model
        df_bullish = df[bull_features]
        df_bearish = df[bear_features]
        
        # load ai model 
        bullish_model = joblib.load(f"models/symbolSelection/medTermBullishMarketStatus.joblib")
        bearish_model = joblib.load(f"models/symbolSelection/medTermBearishMarketStatus.joblib")

        # get most recent data to predict
        bullish_entry = df_bullish.iloc[-1]
        bearish_entry = df_bearish.iloc[-1]
        log.debug(f"med term test simulation {cutoff_date}: associated index with data: {df.index[-1]}")

        # model expect 2d array of entry
        bullish_entry_arr = bullish_entry.values.reshape(1, -1)
        bearish_entry_arr = bearish_entry.values.reshape(1, -1)

        # make prediction
        bull_prediction = int(bullish_model.predict(bullish_entry_arr))
        bear_prediction = int(bearish_model.predict(bearish_entry_arr))
        log.debug(f"{cutoff_date} med term bull prediction: {bull_prediction} | bear prediction: {bear_prediction}")

        # delete macro data - no longer needed
        MonthlyMacroData.objects.all().delete()
        QuarterlyMacroData.objects.all().delete()
        AnnualMacroData.objects.all().delete()

        # bearish models are inconsistent - if precision is bad, overwrite prediction to 0
        if neg_prec < config.models.minPrecision:
            log.info(f"med term bearish model precision too low: {neg_prec} - overwriting prediction")
            bear_prediction = 0

        log.debug(f"{cutoff_date} med term atr spike value: {df['atr_spike'][-1]} and date: {df.index[-1]}")
        if df['atr_spike'][-1] == 0:
            log.info(f"{cutoff_date} med term latest atr spike is 0 - defaulting bearish prediction to 0")
            bear_prediction = 0

        # for now, log feature importance on models that predicted 1 - want to see whats causing false positives
        if bull_prediction == 1:
            bull_importances = bullish_model.feature_importances_
            bull_feature_importance_df = pd.DataFrame({
                'features': bull_features,
                'importance': bull_importances,
            }).sort_values(by='importance', ascending=False)
            log.debug(f"med term bullish features importance: {bull_feature_importance_df}")

        if bear_prediction == 1:
            bear_importances = bearish_model.feature_importances_
            bear_feature_importance_df = pd.DataFrame({
                'features': bear_features,
                'importance': bear_importances,
            }).sort_values(by='importance', ascending=False)
            log.debug(f"med term bearish features importance: {bear_feature_importance_df}")

        final_prediction = 0
        if bull_prediction == 1 and bear_prediction == 0:
            final_prediction = 1
        if bull_prediction == 0 and bear_prediction == 1:
            final_prediction = -1
        
        return bull_prediction, bear_prediction, final_prediction
    except Exception as err:
        log.error(f"error getting med-term market status - defaulting to neutral to be safe: {err}", exc_info=True)

        MonthlyMacroData.objects.all().delete()
        QuarterlyMacroData.objects.all().delete()
        AnnualMacroData.objects.all().delete()

        return 0, 0, 0
    
'''
@param cutoff_date - datetime
'''
def get_long_term_market_status(cutoff_date):
    try:
        log.debug(f"getting long-term market status")
        get_macro(start_date=config.start_date)
        _, df_macro = get_macro_from_db(tradingFrame=config.tradingFrame)

        # get latest data before cutoff date
        df_macro = df_macro[df_macro.index <= cutoff_date]
        df_macro = df_macro[['ism_pmi_macro_exp', 'industrial_production_mom_macro_exp', 'cpi_macro_exp', 'unemployment_rate_macro_exp', 'consumer_sentiment_macro_exp', 'ism_pmi_sma4_macro_exp', 'industrial_production_mom_sma4_macro_exp', 'cpi_sma4_macro_exp', 'unemployment_rate_sma4_macro_exp', 'consumer_sentiment_sma4_macro_exp', 'ism_pmi_sma4_diff_macro_exp', 'industrial_production_mom_sma4_diff_macro_exp', 'cpi_sma4_diff_macro_exp', 'unemployment_rate_sma4_diff_macro_exp', 'consumer_sentiment_sma4_diff_macro_exp', 'ism_pmi_diff_macro_exp', 'industrial_production_mom_diff_macro_exp', 'cpi_diff_macro_exp', 'unemployment_rate_diff_macro_exp', 'consumer_sentiment_diff_macro_exp', 'ism_pmi_cross_above_macro_exp', 'industrial_production_mom_cross_above_macro_exp', 'cpi_cross_above_macro_exp', 'unemployment_rate_cross_above_macro_exp', 'consumer_sentiment_cross_above_macro_exp', 'ism_pmi_cross_below_macro_exp', 'industrial_production_mom_cross_below_macro_exp', 'cpi_cross_below_macro_exp', 'unemployment_rate_cross_below_macro_exp', 'consumer_sentiment_cross_below_macro_exp', 'ism_pmi_dvg_above_macro_exp', 'industrial_production_mom_dvg_above_macro_exp', 'cpi_dvg_above_macro_exp', 'unemployment_rate_dvg_above_macro_exp', 'consumer_sentiment_dvg_above_macro_exp', 'ism_pmi_dvg_below_macro_exp', 'industrial_production_mom_dvg_below_macro_exp', 'cpi_dvg_below_macro_exp', 'unemployment_rate_dvg_below_macro_exp', 'consumer_sentiment_dvg_below_macro_exp',]]

        log.info(f"cut off date: {cutoff_date} new macro data index: {df_macro.index}")
        log.debug(f"df macro: {df_macro['ism_pmi_macro_exp']}")

        # split into weekly data - if index is a friday
        df_macro_weekly = df_macro[df_macro.index.weekday == 4]
        log.debug(f"weekly macro df index: {df_macro_weekly.index}")

        # get spy data and indicators 
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        spy_prices = {
            'date': [],
            'close': [],
            'high': [],
            'low': [],
        }
        response = api.get_bars(symbol='SPY', start=config.start_date, end=cutoff_date.replace(tzinfo=None).date(), timeframe='1Day', limit=10000)
        log.debug(f"SPY data first index: {response[0].t} | final index: {response[-1].t} | cutoff date: {cutoff_date}")
        for result in response:
            spy_prices['date'].append(result.t.to_pydatetime())
            spy_prices['close'].append(float(result.c))
            spy_prices['high'].append(float(result.h))
            spy_prices['low'].append(float(result.l))
        
        spy_df = pd.DataFrame(data=spy_prices)
        spy_df.set_index('date', inplace=True)
        spy_df.index = spy_df.index.tz_convert('US/Eastern')

        # price data
        spy_df['spy_sma50'] = ta.sma(spy_df['close'], length=50)
        spy_df['spy_sma200'] = ta.sma(spy_df['close'], length=200)

        spy_df['spy_sma50_diff'] = spy_df['close'] - spy_df['spy_sma50']
        spy_df['spy_sma200_diff'] = spy_df['close'] - spy_df['spy_sma200']

        prev_price = spy_df['close'].shift(1)
        prev_sma50 = spy_df['spy_sma50'].shift(1)
        prev_sma200 = spy_df['spy_sma200'].shift(1)

        spy_df['spy_sma50_cross_above'] = ((spy_df['close'] > spy_df['spy_sma50']) & (prev_price <= prev_sma50)).astype(int)
        spy_df['spy_sma200_cross_above'] = ((spy_df['close'] > spy_df['spy_sma200']) & (prev_price <= prev_sma200)).astype(int)
        spy_df['spy_sma50_cross_below'] = ((spy_df['close'] < spy_df['spy_sma50']) & (prev_price >= prev_sma50)).astype(int)
        spy_df['spy_sma200_cross_below'] = ((spy_df['close'] < spy_df['spy_sma200']) & (prev_price >= prev_sma200)).astype(int)

        spy_df['std50'] = ta.stdev(spy_df['close'], length=50)
        spy_df['std200'] = ta.stdev(spy_df['close'], length=200)

        spy_df['spy_sma50_dvg_above'] = (spy_df['close'] > (spy_df['spy_sma50'] + spy_df['std50'])).astype(int)
        spy_df['spy_sma200_dvg_above'] = (spy_df['close'] > (spy_df['spy_sma200'] + spy_df['std200'])).astype(int)
        spy_df['spy_sma50_dvg_below'] = (spy_df['close'] < (spy_df['spy_sma50'] - spy_df['std50'])).astype(int)
        spy_df['spy_sma200_dvg_below'] = (spy_df['close'] < (spy_df['spy_sma200'] - spy_df['std200'])).astype(int)

        spy_df.drop(columns=['std50', 'std200'], inplace=True)

        # atr data
        spy_df['atr'] = ta.atr(high=spy_df['high'], low=spy_df['low'], close=spy_df['close'], length=14)

        atr_sma10 = ta.sma(spy_df['atr'], length=10)
        spy_df['atr_spike'] = (spy_df['atr'] > (1.1 * atr_sma10)).astype(int)
        spy_df['atr_spike_smoothed'] = spy_df['atr'] / atr_sma10
        spy_df['atr_sma50'] = ta.sma(spy_df['atr'], length=50)

        spy_df['atr_sma50_diff'] = spy_df['atr'] - spy_df['atr_sma50']

        prev_atr = spy_df['atr'].shift(1)
        prev_atr_sma50 = spy_df['atr_sma50'].shift(1)

        spy_df['atr_sma50_cross_above'] = ((spy_df['atr'] > spy_df['atr_sma50']) & (prev_atr <= prev_atr_sma50)).astype(int)
        spy_df['atr_sma50_cross_below'] = ((spy_df['atr'] < spy_df['atr_sma50']) & (prev_atr >= prev_atr_sma50)).astype(int)

        spy_df['std50'] = ta.stdev(spy_df['atr'], length=50)

        spy_df['atr_sma50_dvg_above'] = (spy_df['atr'] > (spy_df['atr_sma50'] + spy_df['std50'])).astype(int)
        spy_df['atr_sma50_dvg_below'] = (spy_df['atr'] < (spy_df['atr_sma50'] - spy_df['std50'])).astype(int)

        spy_df.drop(columns=['high', 'low', 'std50'], inplace=True) # no longer needed

        spy_df = spy_df[spy_df.index.weekday == 4]
        log.debug(f"spy df new index: {spy_df.index} | length: {len(spy_df.index)}")
        
        # get y-vars - if close increased 10 trading days later = 1, 0 otherwise
        pos_pct_change_20d = [0] * len(spy_df['close'])
        total_bulls = 0
        neg_pct_change_20d = [0] * len(spy_df['close'])
        total_bears = 0
        for i in range(len(spy_df['close']) - 4):
            pct_change = (spy_df['close'].iloc[i + 4] - spy_df['close'].iloc[i]) / np.abs(spy_df['close'].iloc[i])
            if pct_change > config.stocks.long_term_market.pos_pct_change:
                pos_pct_change_20d[i] = 1
                total_bulls += 1

            if pct_change < config.stocks.long_term_market.neg_pct_change:
                neg_pct_change_20d[i] = 1
                total_bears += 1 
        log.debug(f"{cutoff_date} long term total training bulls: {total_bulls} | total training bears: {total_bears}")
        spy_df['pos_pct_change_20d'] = pd.Series(pos_pct_change_20d, index=spy_df.index)
        spy_df['neg_pct_change_20d'] = pd.Series(neg_pct_change_20d, index=spy_df.index)
        # spy_df.drop(columns=['close'], inplace=True) # no longer needed

        log.debug(f"{cutoff_date} long term number of training bearish responses: {sum(spy_df['neg_pct_change_20d'])} | number of training atr spikes: {sum(spy_df['atr_spike'])}")
        for i in range(len(spy_df.index)):
            if spy_df['neg_pct_change_20d'][i] == 1 and spy_df['atr_spike'][i] == 1:
                log.debug(f"long term both bearish response and atr spike are 1 on {spy_df.index[i]}")


        # merge data 
        log.debug(f"weekly macro index: {df_macro_weekly.index}")
        log.debug(f"weekly spy index: {spy_df.index}")
        df = df_macro_weekly.join(spy_df, how='inner')
        log.debug(f"joined df index: {df.index}")
        df_truncated = df.copy()

        # dont want latest date to be in trained model - we are predicting this
        df_truncated = df_truncated[df.index < cutoff_date]
        log.debug(f"df_truncated last index for {cutoff_date} prediction: {df_truncated.index[-1]}")

        # train model
        features = df_truncated.columns.tolist()
        features.remove('pos_pct_change_20d')
        features.remove('neg_pct_change_20d')
        log.debug(f"features: {features}")

        # get best feature combination for each 
        num_features = len(features)
        log.debug(f"number of total features for long term models: {num_features}")

        # create bullish model
        bull_features, bull_prec, bull_acc = get_best_feature_combination(df=df_truncated.drop(columns=['neg_pct_change_20d']), signal_col_name='pos_pct_change_20d', training_fxn=seq_train_model, balanced=False, num_combs=1000, min_comb_len=int(0.25 * num_features), max_comb_len=int(0.5 * num_features))
        log.debug(f"{cutoff_date} long term best bull precision: {bull_prec} | acc: {bull_acc} | features: {bull_features}")
        if bull_features != []: # save model if successful features found
            _, pos_acc, pos_prec, _ = seq_train_model(df=df_truncated, features=bull_features, target_y='pos_pct_change_20d', balanced=False, fileName="symbolSelection/longTermBullishMarketStatus")
            log.debug(f"long term precision, accuracy and features for predicting bullish market status using sequential sampling: {pos_prec} | {pos_acc} | {bull_features}")
        else: # if bull model failed, revert to random training
            log.warning(f"long term no strong model returned for bullish - reverting to random training")
            bull_features, bull_prec, bull_acc = get_best_feature_combination(df=df_truncated.drop(columns=['neg_pct_change_20d']), signal_col_name='pos_pct_change_20d', training_fxn=train_model, balanced=False, num_combs=1000, min_comb_len=int(0.25 * num_features), max_comb_len=int(0.5 * num_features))
            log.debug(f"{cutoff_date} long term best bull precision using random sampling: {bull_prec} | acc: {bull_acc} | features: {bull_features}")
            if bull_features == []:
                log.warning(f"long term no strong model returned for bullish - reverting to all features")
                bull_features = features 

            _, pos_acc, pos_prec, _ = train_model(df=df_truncated, features=bull_features, target_y='pos_pct_change_20d', balanced=False, fileName='symbolSelection/longTermBullishMarketStatus')
            log.debug(f"long term precision, accuracy and features for predicting bullish market status using random sampling: {pos_prec} | {pos_acc} | {bull_features}")

        # create bearish model
        bear_features, bear_prec, bear_acc = get_best_feature_combination(df=df_truncated.drop(columns=['pos_pct_change_20d']), signal_col_name='neg_pct_change_20d', training_fxn=seq_train_model, balanced=False, num_combs=2000, min_comb_len=int(0.25 * num_features), max_comb_len=int(0.5 * num_features))
        log.debug(f"{cutoff_date} long term best bear precision: {bear_prec} | acc: {bear_acc} | features: {bear_features}")
        if bear_features != []: # save model if successful features found
            _, neg_acc, neg_prec, _ = seq_train_model(df=df_truncated, features=bear_features, target_y='neg_pct_change_20d', balanced=False, fileName='symbolSelection/longTermBearishMarketStatus')
            log.debug(f"long term precision, accuracy and features for predicting bearish market status using sequential sampling: {neg_prec} | {neg_acc} | {bear_features}")
        else: # if bear model failed, revert to random training
            log.warning(f"long term no strong model returned for bearish - reverting to random training")
            bear_features, bear_prec, bear_acc = get_best_feature_combination(df=df_truncated.drop(columns=['pos_pct_change_20d']), signal_col_name='neg_pct_change_20d', training_fxn=train_model, balanced=False, num_combs=1000, min_comb_len=int(0.25 * num_features), max_comb_len=int(0.5 * num_features))
            log.debug(f"{cutoff_date} long term best bear precision using random sampling: {bear_prec} | acc: {bear_acc} | features: {bear_features}")
            if bear_features == []:
                log.warning(f"long term no strong model returned for bearish - reverting to all features")
                bear_features = features 

            _, neg_acc, neg_prec, _ = train_model(df=df_truncated, features=bear_features, target_y='neg_pct_change_20d', balanced=False, fileName='symbolSelection/longTermBearishMarketStatus')
            log.debug(f"long term precision, accuracy and features for predicting bearish market status using random sampling: {neg_prec} | {neg_acc} | {bear_features}")

        # load latest data and predict next weeks market status
        # get features we need for this model
        df_bullish = df[bull_features]
        df_bearish = df[bear_features]
        
        # load ai model 
        bullish_model = joblib.load(f"models/symbolSelection/longTermBullishMarketStatus.joblib")
        bearish_model = joblib.load(f"models/symbolSelection/longTermBearishMarketStatus.joblib")

        # get most recent data to predict
        bullish_entry = df_bullish.iloc[-1]
        bearish_entry = df_bearish.iloc[-1]
        log.debug(f"long term test simulation {cutoff_date}: associated index with data: {df.index[-1]}")

        # model expect 2d array of entry
        bullish_entry_arr = bullish_entry.values.reshape(1, -1)
        bearish_entry_arr = bearish_entry.values.reshape(1, -1)

        # make prediction
        bull_prediction = int(bullish_model.predict(bullish_entry_arr))
        bear_prediction = int(bearish_model.predict(bearish_entry_arr))
        log.debug(f"{cutoff_date} long term bull prediction: {bull_prediction} | bear prediction: {bear_prediction}")

        # delete macro data - no longer needed
        MonthlyMacroData.objects.all().delete()
        QuarterlyMacroData.objects.all().delete()
        AnnualMacroData.objects.all().delete()

        # bearish models are inconsistent - if precision is bad, overwrite prediction to 0
        if neg_prec < config.models.minPrecision:
            log.info(f"long term bearish model precision too low: {neg_prec} - overwriting prediction")
            bear_prediction = 0

        log.debug(f"{cutoff_date} long term atr spike value: {df['atr_spike'][-1]} and date: {df.index[-1]}")
        if df['atr_spike'][-1] == 0:
            log.info(f"{cutoff_date} long term latest atr spike is 0 - defaulting bearish prediction to 0")
            bear_prediction = 0

        # for now, log feature importance on models that predicted 1 - want to see whats causing false positives
        if bull_prediction == 1:
            bull_importances = bullish_model.feature_importances_
            bull_feature_importance_df = pd.DataFrame({
                'features': bull_features,
                'importance': bull_importances,
            }).sort_values(by='importance', ascending=False)
            log.debug(f"long term bullish features importance: {bull_feature_importance_df}")

        if bear_prediction == 1:
            bear_importances = bearish_model.feature_importances_
            bear_feature_importance_df = pd.DataFrame({
                'features': bear_features,
                'importance': bear_importances,
            }).sort_values(by='importance', ascending=False)
            log.debug(f"long term bearish features importance: {bear_feature_importance_df}")

        final_prediction = 0
        if bull_prediction == 1 and bear_prediction == 0:
            final_prediction = 1
        if bull_prediction == 0 and bear_prediction == 1:
            final_prediction = -1
        
        return bull_prediction, bear_prediction, final_prediction
    except Exception as err:
        log.error(f"error getting long term market status - defaulting to neutral to be safe: {err}", exc_info=True)

        MonthlyMacroData.objects.all().delete()
        QuarterlyMacroData.objects.all().delete()
        AnnualMacroData.objects.all().delete()

        return 0, 0, 0
'''
given a end_date and a timeframe, returns a list of ad ratios for each timeframe 
@param retry_attempts - int
@param end_date - datetime
@param timeframe - string - daily or weekly'''
def get_ad_ratio(retry_attempts, end_date, timeframe):
    try:
        # working with fridays - temporary fix
        while end_date.weekday() != 4:
            log.debug(f"end date weekday: {end_date.weekday()}")
            end_date = end_date - timedelta(days=1)

        start_date = config.start_date

        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        ad_ratio_query = "SELECT * FROM core_adratios;"
        df_db = pd.read_sql(ad_ratio_query, conn)
        conn.close()
        df_db.drop(columns=['uid'], inplace=True)

        if len(df_db) != 0:
            df_db['date'] = pd.to_datetime(df_db['date'], utc=True)
            df_db.set_index('date', inplace=True)
            df_db.index = df_db.index.tz_convert('US/Eastern')

        db_dates = set(df_db.index)

        # make api call to get most symbols in us
        client = finnhub.Client(api_key=config.finnhub.apikey)
        mics = ['XNYS','XASE','BATS','XNAS']
        symbols = []
        for mic in mics:
            # get list of symbols
            try:
                response = client.stock_symbols(exchange='US', mic=mic, currency='USD')
            except Exception:
                if retry_attempts > 0:
                    log.warning(f"error getting list of symbols - retry attempts left: {retry_attempts}")
                    time.sleep(10)
                    return get_ad_ratio(retry_attempts=retry_attempts - 1, end_date=end_date, timeframe=timeframe)
                else:
                    log.error(f"error: unable to get symbols")
                    return None
                
            for stock in response:
                symbols.append(stock.get('symbol'))

        # remove any stocks with periods - these are substocks
        symbols = list(filter(lambda x: "." not in x, symbols))

        advanced_count = 0
        declined_count = 0

        # alpaca api
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')
        
        # for each symbol, get all prices and add to list in dictionary
        symbol_prices = {}

        # get spy data first to gauge length
        response = api.get_bars(symbol='SPY', timeframe='1Day', feed='sip', sort='asc', start=start_date, end=end_date.replace(tzinfo=None).date(), limit=10000)
        spy_len = len(response)
        symbol_prices['date'] = []
        for result in response:
            symbol_prices['date'].append(result.t.to_pydatetime())

        # check if final entry already in db - have to do here because its not guaranteed that end_date is the final entry in the api call
        final_date = symbol_prices['date'][-1]
        log.debug(f"final spy date: {final_date}")
        if final_date in db_dates:
            log.info(f"data already present - returning db df")
            df_db = df_db[df_db.index <= end_date]
            log.info(f"final df_db index: {df_db.index}")
            return df_db

        for symbol in symbols:
            response = api.get_bars(symbol=symbol, timeframe='1Day', feed='sip', sort='asc', start=start_date, end=end_date.replace(tzinfo=None).date(), limit=10000)
            time.sleep(0.2)
            if len(response) == spy_len:
                symbol_prices[symbol] = []
                for result in response:
                    symbol_prices[symbol].append(float(result.c))

        symbol_prices_df = pd.DataFrame(symbol_prices)
        symbol_prices_df.set_index('date', inplace=True)

        # if timeframe is weekly - save fridays 
        if timeframe == 'weekly':
            symbol_prices_df = symbol_prices_df[symbol_prices_df.index.weekday == 4]
            spy_len = len(symbol_prices_df.index)

        # for each symbol and each index, determine if the current value is greater than or less than the previous value - save final ratio in a list
        advanced_by_date = [0]
        declined_by_date = [0]
        ad_ratio_by_date = [0]
        index = 1
        while index < spy_len:
            advanced_count = 0
            declined_count = 0
            for symbolKey in symbol_prices_df.keys():
                prev_value = symbol_prices_df[symbolKey][index - 1]
                cur_value = symbol_prices_df[symbolKey][index]

                if cur_value > prev_value:
                    advanced_count += 1
                if cur_value < prev_value:
                    declined_count += 1
            log.debug(f"ad ratio for index {symbol_prices_df.index[index]}: {float(advanced_count / declined_count)}")
            advanced_by_date.append(advanced_count)
            declined_by_date.append(declined_count)
            ad_ratio_by_date.append(float(advanced_count / declined_count))
            index += 1

        ad_ratio_series = pd.Series(ad_ratio_by_date, index=symbol_prices_df.index) 
        advanced_series = pd.Series(advanced_by_date, index=symbol_prices_df.index) 
        declined_series = pd.Series(declined_by_date, index=symbol_prices_df.index) 
        symbol_prices_df['ad_ratio'] = ad_ratio_series
        symbol_prices_df['advanced'] = advanced_series
        symbol_prices_df['declined'] = declined_series
        symbol_prices_df = symbol_prices_df[['ad_ratio']]

        for i in range(len(symbol_prices_df['ad_ratio'])):
            if symbol_prices_df.index[i] not in db_dates:
                log.debug(f"new ad ratio on {symbol_prices_df.index[i]}")
                ADRatios(
                    date=symbol_prices_df.index[i],
                    advanced=symbol_prices_df['advanced'][i],
                    declined=symbol_prices_df['declined'][i], 
                    ad_ratio=symbol_prices_df['ad_ratio'][i],
                ).save()
            else:
                log.info(f"ad ratio already present for {symbol_prices_df.index[i]} - skipping")

        return symbol_prices_df
    except Exception as err:
        log.error(f"error getting a/d ratio: {err}", exc_info=True)
        return None
    
''' takes a stock symbol, and a date, and computes the beta value as of that date
    beta equation:
        - get daily stock price data for S&P (SPY) and the stock
        - get the percent change for each day
        - calculate covariance and variance of data
        - divide covariance by variance

    @param symbol - string
    @param end_date - datetime
'''
def compute_beta(symbol, end_date):
    try:
        # get price data of symbol and SPY
        start_date = end_date - timedelta(days=400)

        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        # get prices for symbol
        stock_prices = []
        response = api.get_bars(symbol=symbol, start=start_date, end=end_date, timeframe='1Day')
        time.sleep(0.1)

        log.debug(f"len of {symbol} data: {len(response)}")
        if len(response) == 0:
            log.warning(f"compute beta: {symbol} price data empty from {start_date} to {end_date} - skipping")
            return None

        for result in response:
            stock_prices.append(float(result.c))

        log.debug(f"compute beta: {symbol} first date: {response[0].t} | last date: {response[-1].t} | length: {len(stock_prices)}")
        stock_prices = pd.Series(stock_prices)

        # get market prices
        spy_prices = []
        response = api.get_bars(symbol='SPY', start=start_date, end=end_date, timeframe='1Day', limit=10000)
        time.sleep(0.1)

        for result in response:
            spy_prices.append(float(result.c))

        log.debug(f"compute beta: SPY first date: {response[0].t} | last date: {response[-1].t} | length: {len(spy_prices)}")
        spy_prices = pd.Series(spy_prices)

        df = pd.DataFrame({
            'stock': stock_prices,
            'market': spy_prices,
        }).dropna()

        # get percent change
        df_pct = my_pct_change(df=df)

        # compute beta
        cov = df_pct['stock'].cov(df_pct['market'])
        var = df_pct['market'].var()

        beta = cov / var
        log.info(f"beta for {symbol} as of {end_date} is {beta}")
        return beta
    except Exception as err:
        log.error(f"error computing beta for {symbol}: {err}", exc_info=True)
        return None
    
'''
@param symbols - list
NOTE: all datetimes should be Eastern in this function
NOTE: consider swapping from daily index to weekly - less noise
TODO: analyze all code to make verify:
    - dates are correct
    - y-vars are correct
'''
def get_stock_data(symbols):
    try:
        # key: symbol, val: dataframe containing all data
        symbol_dict = {}

        # clear db before and after 
        QEarnings.objects.all().delete()
        for symbol in symbols:
            log.debug(f"getting {symbol} data")
            time.sleep(0.1) # slow down to handle rate limit
            start_time = time.time()
            # NOTE: do initial filtering upfront to save time
            # price data
            df = get_stock_metrics(symbol=symbol, retry_attempts=3)

            if df is None:
                log.warning(f"stock metric df not returned for {symbol} - skipping")
                continue
            
            # causing sma 200 issues
            if len(df['close']) < 200:
                log.warning(f"not enough stock price data for {symbol} - skipping")
                continue

            # NOTE: Eastern timezone
            log.debug(f"{symbol} stock price data timezone: {df.index}")

            # earnings data
            log.debug(f"getting earnings for {symbol}")
            get_earnings(symbol=symbol, start_date=config.start_date)
            _, df_earnings = get_earnings_from_db(symbol=symbol, tradingFrame=config.tradingFrame, withTranscripts=False)
            if df_earnings is None:
                log.warning(f"earnings data not available for {symbol} - skipping")
                log.debug(f"time to get {symbol} data: {(time.time() - start_time) / 60} minutes")
                continue
            
            # insider data
            end_date = datetime.today().date()
            get_insider_transactions(symbol=symbol, start_date=config.start_date, end_date=end_date)
            _, df_insider = get_insider_from_db(symbol=symbol, tradingFrame=config.tradingFrame) # left return value is raw data, right is extended daily data

            if df_insider is None:
                log.info(f"{symbol} insider data not available - skipping")
                log.debug(f"time to get {symbol} data: {(time.time() - start_time) / 60} minutes")
                continue

            # merge dataframes
            log.debug(f"stock price df timezone check: {df.index.tz}")
            log.debug(f"q-earnings df timezone check: {df_earnings.index.tz}")
            log.debug(f"insider transaction df timezone check: {df_insider.index.tz}")

            df = df.join(df_earnings, how='inner')
            df = df.join(df_insider, how='inner')
            df.index = df.index.tz_convert('US/Eastern')
            log.debug(f"final joined {symbol} index: {df.index}")

            symbol_dict[symbol] = df
            log.debug(f"time to get {symbol} data: {(time.time() - start_time) / 60} minutes")

        QEarnings.objects.all().delete()
        return symbol_dict
    except Exception as err:
        log.error(f"error getting stock symbols: {err}", exc_info=True)
        QEarnings.objects.all().delete()
        return None
    
def get_stock_indicators(symbol_dataframes):
    try:
        start_time = time.time()

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
        time.sleep(0.1) # rate limit
        for result in response:
            date = result.t.to_pydatetime()
            spy_dict['date'].append(date)
            spy_dict['close'].append(float(result.c))
        spy_df = pd.DataFrame(spy_dict).set_index('date')
        spy_df.index = spy_df.index.tz_convert('US/Eastern')
        
        symbol_dict = {}
        for symbol, df in symbol_dataframes.items():
            spy_df_copy = spy_df.copy()
            # NOTE: if shorter data is leading to inaccurate predictions, remove
            # ideally want data from 2022-01-01, but will accept as early as 2023-01-01
            log.debug(f"SPY df index timezone: {spy_df_copy.index}")
            if len(spy_df_copy['close']) != len(df['close']):
                # data doesnt go back to start of 2022, see if 2023 available
                eastern = pytz.timezone('US/Eastern')
                latest_allowed_dt = eastern.localize(datetime(2023, 1, 1, 0, 0, 0))
                cutoff_dt = df.index.min()

                if cutoff_dt > latest_allowed_dt:
                    log.warning(f"not enough stock price data for {symbol}: data isnt from before 2023 - skipping")
                    log.debug(f"time to get {symbol} data: {(time.time() - start_time) / 60} minutes")
                    continue
                spy_df_copy = spy_df_copy[spy_df_copy.index >= cutoff_dt]

            # compute price indicators 
            df['sma20'] = ta.sma(df['close'], length=20)
            df['sma50'] = ta.sma(df['close'], length=50)
            df['sma200'] = ta.sma(df['close'], length=200)

            df['sma20_diff'] = df['close'] - df['sma20']
            df['sma50_diff'] = df['close'] - df['sma50']
            df['sma200_diff'] = df['close'] - df['sma200']

            df['sma20_mtm'] = df['sma20'].diff()
            df['sma50_mtm'] = df['sma50'].diff()
            df['sma200_mtm'] = df['sma200'].diff()

            prev_close = df['close'].shift(1)

            prev_sma20 = df['sma20'].shift(1)
            df['sma20_cross_above'] = ((df['close'] > df['sma20']) & (prev_close <= prev_sma20)).astype(int)
            df['sma20_cross_below'] = ((df['close'] < df['sma20']) & (prev_close >= prev_sma20)).astype(int)
            
            close_std20 = ta.stdev(df['close'], length=20)
            df['sma20_dvg_above'] = (df['close'] > (df['sma20'] + close_std20)).astype(int)
            df['sma20_dvg_below'] = (df['close'] < (df['sma20'] - close_std20)).astype(int)

            prev_sma50 = df['sma50'].shift(1)
            df['sma50_cross_above'] = ((df['close'] > df['sma50']) & (prev_close <= prev_sma50)).astype(int)
            df['sma50_cross_below'] = ((df['close'] < df['sma50']) & (prev_close >= prev_sma50)).astype(int)
            
            close_std50 = ta.stdev(df['close'], length=50)
            df['sma50_dvg_above'] = (df['close'] > (df['sma50'] + close_std50)).astype(int)
            df['sma50_dvg_below'] = (df['close'] < (df['sma50'] - close_std50)).astype(int)

            prev_sma200 = df['sma200'].shift(1)
            df['sma200_cross_above'] = ((df['close'] > df['sma200']) & (prev_close <= prev_sma200)).astype(int)
            df['sma200_cross_below'] = ((df['close'] < df['sma200']) & (prev_close >= prev_sma200)).astype(int)
            
            close_std200 = ta.stdev(df['close'], length=200)
            df['sma200_dvg_above'] = (df['close'] > (df['sma200'] + close_std200)).astype(int)
            df['sma200_dvg_below'] = (df['close'] < (df['sma200'] - close_std200)).astype(int)

            # atr indicators 
            df['atr'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
            df['atr_sma20'] = ta.sma(df['atr'], length=20)
            df['atr_sma50'] = ta.sma(df['atr'], length=50)
            df['atr_sma200'] = ta.sma(df['atr'], length=200)

            atr_sma10 = ta.sma(df['atr'], length=10)
            df['atr_spike_smoothed'] = df['atr'] / atr_sma10

            df['atr_sma20_diff'] = df['atr'] - df['atr_sma20']
            df['atr_sma50_diff'] = df['atr'] - df['atr_sma50']
            df['atr_sma200_diff'] = df['atr'] - df['atr_sma200']

            df['atr_sma20_mtm'] = df['atr_sma20'].diff()
            df['atr_sma50_mtm'] = df['atr_sma50'].diff()
            df['atr_sma200_mtm'] = df['atr_sma200'].diff()

            prev_atr = df['atr'].shift(1)

            prev_atr_sma20 = df['atr_sma20'].shift(1)
            df['atr_sma20_cross_above'] = ((df['atr'] > df['atr_sma20']) & (prev_atr <= prev_atr_sma20)).astype(int)
            df['atr_sma20_cross_below'] = ((df['atr'] < df['atr_sma20']) & (prev_atr >= prev_atr_sma20)).astype(int)
            
            atr_std20 = ta.stdev(df['atr'], length=20)
            df['atr_sma20_dvg_above'] = (df['atr'] > (df['atr_sma20'] + atr_std20)).astype(int)
            df['atr_sma20_dvg_below'] = (df['atr'] < (df['atr_sma20'] - atr_std20)).astype(int)

            prev_atr_sma50 = df['atr_sma50'].shift(1)
            df['atr_sma50_cross_above'] = ((df['atr'] > df['atr_sma50']) & (prev_atr <= prev_atr_sma50)).astype(int)
            df['atr_sma50_cross_below'] = ((df['atr'] < df['atr_sma50']) & (prev_atr >= prev_atr_sma50)).astype(int)
            
            atr_std50 = ta.stdev(df['atr'], length=50)
            df['atr_sma50_dvg_above'] = (df['atr'] > (df['atr_sma50'] + atr_std50)).astype(int)
            df['atr_sma50_dvg_below'] = (df['atr'] < (df['atr_sma50'] - atr_std50)).astype(int)

            prev_atr_sma200 = df['atr_sma200'].shift(1)
            df['atr_sma200_cross_above'] = ((df['atr'] > df['atr_sma200']) & (prev_atr <= prev_atr_sma200)).astype(int)
            df['atr_sma200_cross_below'] = ((df['atr'] < df['atr_sma200']) & (prev_atr >= prev_atr_sma200)).astype(int)
            
            atr_std200 = ta.stdev(df['atr'], length=200)
            df['atr_sma200_dvg_above'] = (df['atr'] > (df['atr_sma200'] + atr_std200)).astype(int)
            df['atr_sma200_dvg_below'] = (df['atr'] < (df['atr_sma200'] - atr_std200)).astype(int)

            # beta
            symbol_pct_change = df['close'].pct_change()
            spy_pct_change = spy_df_copy['close'].pct_change()

            rolling_cov_60 = symbol_pct_change.rolling(60).cov(spy_pct_change)
            rolling_var_60 = spy_pct_change.rolling(60).var()
            df['beta_60'] = rolling_cov_60 / rolling_var_60

            rolling_cov_200 = symbol_pct_change.rolling(200).cov(spy_pct_change)
            rolling_var_200 = spy_pct_change.rolling(200).var()
            df['beta_200'] = rolling_cov_200 / rolling_var_200

            # performance vs SPY 
            df['spy_symbol_corr10'] = symbol_pct_change.rolling(window=10).corr(spy_pct_change)
            df['spy_symbol_corr30'] = symbol_pct_change.rolling(window=30).corr(spy_pct_change)

            df['rolling_pct_change'] = df['close'].pct_change(periods=10)
            rolling_spy_pct_change = spy_df_copy['close'].pct_change(periods=10)
            df['spy_pct_diff'] = df['rolling_pct_change'] - rolling_spy_pct_change

            outperform_list = []
            outperform_count = 0
            underperform_list = []
            underperform_count = 0
            gain_streak_list = []
            gain_streak = 0
            loss_streak_list = []
            loss_streak = 0
            prev_symbol_pct_change = symbol_pct_change.shift(1)
            for i in range(len(symbol_pct_change)):
                # outperform spy
                if symbol_pct_change[i] > spy_pct_change[i]:
                    outperform_count += 1
                else: 
                    outperform_count = 0
                outperform_list.append(outperform_count)

                # underperform spy
                if symbol_pct_change[i] < spy_pct_change[i]:
                    underperform_count += 1
                else:
                    underperform_count = 0
                underperform_list.append(underperform_count)

                # positive days in a row 
                if symbol_pct_change[i] > prev_symbol_pct_change[i]:
                    gain_streak += 1
                else:
                    gain_streak = 0
                gain_streak_list.append(gain_streak)

                # negative days in a row
                if symbol_pct_change[i] < prev_symbol_pct_change[i]:
                    loss_streak += 1
                else: 
                    loss_streak = 0
                loss_streak_list.append(loss_streak)
            df['outperform_spy_streak'] = pd.Series(outperform_list, index=symbol_pct_change.index)
            df['underperform_spy_streak'] = pd.Series(underperform_list, index=symbol_pct_change.index)
            df['gain_streak'] = pd.Series(gain_streak_list, index=symbol_pct_change.index)
            df['loss_streak'] = pd.Series(loss_streak_list, index=symbol_pct_change.index)

            # rsi indicators 
            df['rsi'] = ta.rsi(close=df['close'], length=14)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_slope'] = slope_series(series=df['rsi'], window=3)

            # volume indicators 
            df['vol_sma20'] = ta.sma(df['volume'], length=20)
            df['vol_spike'] = df['volume'] / df['vol_sma20']
            df['obv'] = ta.obv(close=df['close'], volume=df['volume'])

            # bollinger bands 
            bbands = ta.bbands(df['close'], length=20, std=2.0)
            df['bband_width'] = bbands[f'BBB_20_2.0']

            # interaction indicators between groups
            df['market_cap'] = df['close'] * df['shares_outstanding_qearnings_exp']

            sma20_slope = df['sma20'].diff(3)
            df['insider_divergence_selloff'] = ((sma20_slope > 0) & (df['advc_long_insider'] < 0)).astype(int)
            df['insider_divergence_buyup'] = ((sma20_slope < 0) & (df['advc_long_insider'] > 0)).astype(int)
            
            sma20_volume = ta.sma(df['volume'], length=20)
            df['divergence_volume_weakening'] = ((df['sma20'] > df['sma50']) & (df['volume'] < sma20_volume)).astype(int)
            df['divergence_volume_strengthening'] = ((df['sma20'] < df['sma50']) & (df['volume'] > sma20_volume)).astype(int)

            df['divergence_rsi_d_reverse'] = ((df['sma20'] > df['sma50']) & (df['rsi_slope'] < 0)).astype(int)
            df['divergence_rsi_u_reverse'] = ((df['sma20'] < df['sma50']) & (df['rsi_slope'] > 0)).astype(int)

            df['divergence_revenue_weakening'] = ((df['sma20'] > df['sma50']) & (df['revenue_diff_qearnings_exp'] < 0)).astype(int)
            df['divergence_revenue_strengthening'] = ((df['sma20'] < df['sma50']) & (df['revenue_diff_qearnings_exp'] > 0)).astype(int)
            
            symbol_dict[symbol] = df
            log.debug(f"adding {symbol} - time to get {symbol} data: {(time.time() - start_time) / 60} minutes")

        return symbol_dict
    except Exception as err:
        log.error(f"error getting stock indicators: {err}", exc_info=True)
'''
@param symbol_dataframes - dictionary: key: symbol, val: dataframe
@param predicted_market_data - dict 
    - keys: 
        - market_status: str
        - timeframe: int
        - pct_change: float
@param cutoff_date - datetime
'''
def analyze_stocks(symbol_dataframes, predicted_market_data, cutoff_date):
    try:    
        chosen_symbols = {} # key: symbol, val: target pct change
        predicted_market_status = predicted_market_data['market_status']
        timeframe = predicted_market_data['timeframe']
        market_pos_pct_change = predicted_market_data['pct_change']
        market_neg_pct_change = -1 * market_pos_pct_change

        # get spy data for market regimes
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
        time.sleep(0.1) # rate limit
        for result in response:
            date = result.t.to_pydatetime()
            spy_dict['date'].append(date)
            spy_dict['close'].append(float(result.c))
        spy_df = pd.DataFrame(spy_dict).set_index('date')
        spy_df.index = spy_df.index.tz_convert('US/Eastern')

        # get historical market regimes - use the same method for macro backtest - determine if spy increase/decreased by x within the next k days
        spy_df = spy_df[['close']].copy()
        spy_df_weekly = spy_df[spy_df.index.weekday == 4]
        market_statuses = [0] * len(spy_df_weekly['close'])
        step_size = int(timeframe / 5) # number of trading weeks 
        for i in range(len(spy_df_weekly['close']) - step_size):
            cur_spy_pct_change = (spy_df_weekly['close'].iloc[i + step_size] - spy_df_weekly['close'].iloc[i]) / np.abs(spy_df_weekly['close'].iloc[i])
            if cur_spy_pct_change > market_pos_pct_change:
                market_statuses[i] = 1
            elif cur_spy_pct_change < market_neg_pct_change:
                market_statuses[i] = -1

        spy_df_weekly['market_status'] = pd.Series(market_statuses, index=spy_df_weekly.index)
        spy_df_weekly.drop(columns=['close'], inplace=True)
        log.debug(f"analyze_stocks {cutoff_date}: spy weekly market df: {spy_df_weekly}")

        # back fill back into daily index
        spy_df_start_time = spy_df_weekly.index.min()
        current_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
        current_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        spy_df_end_time = pd.Timestamp(current_day)
        new_time_index = pd.date_range(start=spy_df_start_time, end=spy_df_end_time, freq='D')

        market_statuses_df = spy_df_weekly.reindex(new_time_index)
        market_statuses_df = market_statuses_df.bfill()
        log.debug(f"analyze_stocks {cutoff_date}: market status backfilled df: {market_statuses_df}")

        for symbol, df in symbol_dataframes.items():
            start_time = time.time()
            log.debug(f"analyze_stocks {cutoff_date}: {cutoff_date} analyzing {symbol}")

            df = df[df.index <= cutoff_date]
            latest_beta = df['beta_60'].loc[cutoff_date]
            if latest_beta < config.stocks.selection.bullish.min_beta or latest_beta > config.stocks.selection.bullish.max_beta:
                log.debug(f"analyze_stocks {cutoff_date}: beta not in threshold for {symbol}: {latest_beta} | min: {config.stocks.selection.bullish.min_beta} | max: {config.stocks.selection.bullish.max_beta}")
                log.debug(f"analyze_stocks {cutoff_date}: time to analyze {symbol}: {(time.time() - start_time) / 60} minutes")
                continue
            latest_market_cap = df['market_cap'].loc[cutoff_date]
            if latest_market_cap * 1000000 < config.stocks.minCap:
                log.debug(f"analyze_stocks {cutoff_date}: market cap too low for {symbol}: {latest_market_cap * 1000000} | required: {config.stocks.minCap}")
                log.debug(f"analyze_stocks {cutoff_date}: time to analyze {symbol}: {(time.time() - start_time) / 60} minutes")
                continue

            # get buy/sell instances series using pct changes 
            if predicted_market_status == 'bullish':
                buy_signals = [0] * len(df['close'])
                pct_changes = float(config.stocks.selection.bullish.long_atr) * (df['atr'] / df['close'])
                for i in range(len(df.index) - timeframe):
                    # log.debug(f"{cutoff_date} - {symbol} - {predicted_market_status} - {timeframe} timeframe-pct change: {((df['close'][i + timeframe] - df['close'][i]) / df['close'][i])} | target atr pct change: {pct_changes[i]}")
                    if ((df['close'][i + timeframe] - df['close'][i]) / df['close'][i]) >= pct_changes[i]:
                        buy_signals[i] = 1
                log.debug(f"analyze_stocks {cutoff_date}: {cutoff_date} - {symbol} - {predicted_market_status} - {timeframe} number of buy signals to train off of {sum(buy_signals)} | total days {len(buy_signals)}")
                df['trade_signals'] = pd.Series(buy_signals, index=df.index)

            elif predicted_market_status == 'bearish':
                pct_changes = float(config.stocks.selection.bearish.short_atr) * (df['atr'] / df['close'])
            else:
                log.warning(f"analyze_stocks {cutoff_date}: neutral market status not implemented yet - skipping")
                return
 
            df = df.join(market_statuses_df, how='inner')
            df.to_csv(f'stockSelectionData/{symbol}_{cutoff_date}_selection.csv', index=True)

            # load data and create model

            # training df
            df_truncated = df.copy()

            # filter to include only dates that are in the expected market status 
            if predicted_market_status == 'bullish':
                df_truncated = df_truncated[df_truncated['market_status'] == 1]
            else: 
                log.error(f"analyze_stocks {cutoff_date}: bearish and neutral markets not implemented yet - skipping")
                return 
            
            df_truncated = df_truncated[df_truncated.index < cutoff_date]
            if len(df_truncated['close']) < 100:
                log.debug(f"analyze_stocks {cutoff_date}: not enough training entries for {symbol} - skipping")
                log.debug(f"analyze_stocks {cutoff_date}: time to analyze {symbol}: {(time.time() - start_time) / 60} minutes")
                continue

            log.debug(f"analyze_stocks {cutoff_date}: number of {predicted_market_status} entries for {symbol} to train with: {len(df_truncated['close'])}")
            log.debug(f"analyze_stocks {cutoff_date}: df_truncated last index for {cutoff_date} prediction: {df_truncated.index[-1]}")

            features = df_truncated.columns.tolist()
            features.remove('trade_signals')
            log.debug(f"analyze_stocks {cutoff_date}: features: {features}")

            # get best feature combination for each 
            num_features = len(features)
            log.debug(f"analyze_stocks {cutoff_date}: number of total features for stock selection model: {num_features}")
            final_features, final_prec, final_acc = get_best_feature_combination(df=df_truncated, signal_col_name='trade_signals', training_fxn=seq_train_model, balanced=False, num_combs=5000, min_comb_len=int(0.04 * num_features), max_comb_len=int(0.08 * num_features), num_threads=10)
            log.debug(f"analyze_stocks {cutoff_date}: {cutoff_date} {symbol} selection best {predicted_market_status} for {timeframe} days precision: {final_prec} | acc: {final_acc} | features: {final_features}")
            if final_features != []: # save model if successful features found
                # TODO: might need to update directories for models for better clarity
                _, acc, prec, _ = seq_train_model(df=df_truncated, features=final_features, target_y='trade_signals', balanced=False, fileName=f"symbolSelection/{predicted_market_status}/{symbol}")
                log.debug(f"analyze_stocks {cutoff_date}: {cutoff_date} - {symbol} - {predicted_market_status} - {timeframe} precision, accuracy and features for predicting bullish movements using sequential sampling: {prec} | {acc} | {final_features}")
            else:
                log.warning(f"{cutoff_date} - {symbol} - {predicted_market_status} - {timeframe} not strong enough - skipping")
                log.debug(f"analyze_stocks {cutoff_date}: time to analyze {symbol}: {(time.time() - start_time) / 60} minutes")
                continue

            # load model and make prediction
            # get features we need for this model
            df = df[final_features]
            
            # load ai model 
            model = joblib.load(f"models/symbolSelection/{predicted_market_status}/{symbol}.joblib")

            # get most recent data to predict
            entry = df.iloc[-1]
            log.debug(f"analyze_stocks {cutoff_date}: stock symbol selection {cutoff_date}: associated index with data: {df.index[-1]} | percent change we are predicting: {pct_changes.loc[df.index[-1]]}")

            # model expect 2d array of entry
            entry_arr = entry.values.reshape(1, -1)

            # make prediction
            prediction = int(model.predict(entry_arr))
            log.debug(f"analyze_stocks {cutoff_date}: {cutoff_date} {symbol} {predicted_market_status} {timeframe} prediction: {prediction}")

            if prediction != 1:
                log.debug(f"analyze_stocks {cutoff_date}: time to analyze {symbol}: {(time.time() - start_time) / 60} minutes")
                continue 
        
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'features': final_features,
                'importance': importances,
            }).sort_values(by='importance', ascending=False)
            log.debug(f"{cutoff_date} - {symbol} - {predicted_market_status} - {timeframe} features importance: {feature_importance_df}")

            if predicted_market_status == 'bullish': 
                log.debug(f"analyze_stocks {cutoff_date}: adding long symbol: {symbol}")
                # ensure_connection() # ensure mysql connection
                # LongStocks(
                #     symbol=symbol,
                # ).save()

                # add results to chosen_symbols dict for backtesting
                log.debug(f"analyze_stocks {cutoff_date}: {cutoff_date} - {symbol} - {predicted_market_status} - {timeframe} associated index and percent change we are predicting: {df.index[-1]} - {pct_changes.loc[df.index[-1]]}")
                chosen_symbols[symbol] = pct_changes.loc[df.index[-1]]
                # TODO: save df to csv file to analyze data
            else:
                log.warning(f"analyze_stocks {cutoff_date}: bearish and neutral not implemented yet")

            log.debug(f"analyze_stocks {cutoff_date}: time to analyze {symbol}: {(time.time() - start_time) / 60} minutes")
            
        return chosen_symbols
    except Exception as err:
        log.error(f"error analyzing stocks: {err}", exc_info=True)
        return
'''
NOTE: data retrieved from API is Eastern timezone - gets converted to UTC by db when stored
'''
def get_stock_metrics(symbol, retry_attempts):
    try:
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=config.alpaca.marketUrl, api_version='v2')
        start_date = datetime(year=2018, month=1, day=1).date() # setting start date super early to account for any rolling calcs later on
        end_date = datetime.today().date() - timedelta(days=5) # because of free-tier alpaca subscription, cant get latest data so need to subtract time

        # stock price data
        symbol_dict = {
            'date': [],
            'open': [],
            'close': [],
            'high': [],
            'low': [],
            'volume': [],
        }
        try: 
            response = api.get_bars(symbol=symbol, timeframe='1Day', feed='sip', sort='asc', adjustment="split", start=start_date, end=end_date, limit=10000)
        except Exception as err:
            if retry_attempts > 0:
                log.error(f"error getting stock price data for {symbol} - retrying")
                return get_stock_metrics(symbol=symbol, retry_attempts=(retry_attempts - 1))
            else:
                log.error(f"error getting stock price data for {symbol}: {err}", exc_info=True)
                return 
        time.sleep(0.1)

        for result in response:
            date = result.t.to_pydatetime()
            symbol_dict['date'].append(date)
            symbol_dict['open'].append(float(result.o))
            symbol_dict['close'].append(float(result.c))
            symbol_dict['high'].append(float(result.h))
            symbol_dict['low'].append(float(result.l))
            symbol_dict['volume'].append(float(result.v))

        symbol_df = pd.DataFrame(symbol_dict).set_index('date')

        if len(symbol_df.index) == 0:
            log.debug(f"{symbol} stock price data empty - skipping")
            return None

        symbol_df.index = symbol_df.index.tz_convert('US/Eastern')
        log.debug(f"get_stock_metrics: stock price df timezone: {symbol_df.index}")

        # convert nans to Nones
        symbol_df.replace({np.nan: None}, inplace=True)

        # lazy so deleting all data before saving 
        # StockMetrics.objects.filter(symbol=symbol).delete()
        # for i in range(len(symbol_df.index)):
        #     StockMetrics(
        #         symbol=symbol, 
        #         date=symbol_df.index[i],
        #         timeframe='1Day',

        #         open=symbol_df['open'][i],
        #         close=symbol_df['close'][i],
        #         high=symbol_df['high'][i],
        #         low=symbol_df['low'][i],
        #         volume=symbol_df['volume'][i],
        #     ).save()

        return symbol_df
    except Exception as err:
        log.error(f"error: unable to get stock metrics for {symbol}: {err}", exc_info=True)
        return None
    
'''
@param symbol - str
@param series - pd series containing stock metrics
@param market_status - str
@param minCap - int
'''
def analyze_short_symbol(symbol, series, market_status, minCap):
    try:
        # variables depending on market status
        status_vars = {
            'bullish': {
                'rsiLow': -100, # no limit
                'rsiHigh': 50, 
                'minBeta': 0.5,
                'maxBeta': 1.2,
            }, 
            'neutral': {
                'rsiLow': 40,
                'rsiHigh': 60,
                'minBeta': 0.8,
                'maxBeta': 1.2,
            },
            'bearish': {
                'rsiLow': 35,
                'rsiHigh': 60,
                'minBeta': 0.0,
                'maxBeta': 0.7,
            }
        }

        rsiHigh = status_vars[market_status]['rsiHigh']
        rsiLow = status_vars[market_status]['rsiLow']
        minBeta = status_vars[market_status]['minBeta']
        maxBeta = status_vars[market_status]['maxBeta']

        beta = series['beta']
        smaLow = series['smaLow']
        smaHigh = series['smaHigh']
        rsi = series['rsi']

        dollar_volume_30 = series['dollar_volume_30']
        dollar_volume_90 = series['dollar_volume_90']

        market_cap = series['market_cap']

        eps = series['eps']
        eps_sma = series['eps_sma']

        net_income = series['net_income']
        net_income_sma = series['net_income_sma']
        
        revenue = series['revenue']
        revenue_sma = series['revenue_sma']

        gross_margin = series['gross_margin']
        gross_margin_sma = series['gross_margin_sma']

        ebit = series['ebit']
        ebit_sma = series['ebit_sma']

        free_cash_flow = series['free_cash_flow']
        free_cash_flow_sma = series['free_cash_flow_sma']

        roe = series['roe']
        roe_sma = series['roe_sma']

        if not beta or not smaLow or not smaHigh or not rsi or not dollar_volume_30 or not dollar_volume_90 or not market_cap:
            log.warning(f"stock price data missing for {symbol} - skipping")
            return
        
        log.debug(f"short: {symbol} beta: {beta} | smaLow: {smaLow} | smaHigh: {smaHigh} | rsi: {rsi}")
        if (beta > minBeta and beta < maxBeta and smaLow < smaHigh and rsi < rsiHigh and rsi > rsiLow):
            log.info(f"short: analyzing insider data")

            log.debug(f"short: {symbol} dv30: {dollar_volume_30} | dv90: {dollar_volume_90}")
            if dollar_volume_30 is None or dollar_volume_90 is None:
                log.info(f"short: {symbol} dollar volume change data not available - skipping")
                return
            
            if dollar_volume_30 < 0 and dollar_volume_30 < (0.9 * dollar_volume_90):
                log.info(f"short: analyzing earnings data")
                if market_cap*1000000 < minCap:
                    log.warning(f"market cap too low for {symbol} - skipping")
                    return
                
                earnings_score = 0

                if eps and eps_sma:
                    if eps < eps_sma:
                        earnings_score += 1
                if net_income and net_income_sma:
                    if net_income < net_income_sma:
                        earnings_score += 1
                if revenue and revenue_sma:
                    if revenue < revenue_sma:
                        earnings_score += 1
                if gross_margin and gross_margin_sma:
                    if gross_margin < gross_margin_sma:
                        earnings_score += 1
                if ebit and ebit_sma:
                    if ebit < ebit_sma:
                        earnings_score += 1
                if free_cash_flow and free_cash_flow_sma:
                    if free_cash_flow < free_cash_flow_sma:
                        earnings_score += 1
                if roe and roe_sma:
                    if roe < roe_sma:
                        earnings_score += 1
                
                log.debug(f"short: final earnings score for {symbol}: {earnings_score}")

                if earnings_score > 3:
                    log.debug(f"adding short symbol: {symbol}")
                    ensure_connection() # ensure mysql connection
                    ShortStocks(
                        symbol=symbol,
                        beta=beta,
                        market_cap=market_cap,
                    ).save()
        return
    except Exception as err:
        log.error(f"error analyzing short symbol {symbol}: {err}", exc_info=True)
        return
    

'''
@param symbol - str
@param series - pd series containing stock metrics
@param market_status - str
@param minCap - int
'''
def analyze_long_symbol(symbol, series, market_status, minCap):
    try:
        # variables depending on market status
        status_vars = {
            'bullish': {
                'rsiLow': 50,
                'rsiHigh': 999, # no limit
                'minBeta': 1.2, 
                'maxBeta': 2.5,
            }, 
            'neutral': {
                'rsiLow': 40,
                'rsiHigh': 60,
                'minBeta': 0.8,
                'maxBeta': 1.3,
            },
            'bearish': {
                'rsiLow': 25,
                'rsiHigh': 40,
                'minBeta': 0.0,
                'maxBeta': 0.8,
            }
        }

        rsiHigh = status_vars[market_status]['rsiHigh']
        rsiLow = status_vars[market_status]['rsiLow']
        minBeta = status_vars[market_status]['minBeta']
        maxBeta = status_vars[market_status]['maxBeta']

        beta = series['beta']
        smaLow = series['smaLow']
        smaHigh = series['smaHigh']
        rsi = series['rsi']

        dollar_volume_30 = series['dollar_volume_30']
        dollar_volume_90 = series['dollar_volume_90']

        market_cap = series['market_cap']

        eps = series['eps']
        eps_sma = series['eps_sma']

        net_income = series['net_income']
        net_income_sma = series['net_income_sma']
        
        revenue = series['revenue']
        revenue_sma = series['revenue_sma']

        gross_margin = series['gross_margin']
        gross_margin_sma = series['gross_margin_sma']

        ebit = series['ebit']
        ebit_sma = series['ebit_sma']

        free_cash_flow = series['free_cash_flow']
        free_cash_flow_sma = series['free_cash_flow_sma']

        roe = series['roe']
        roe_sma = series['roe_sma']

        if not beta or not smaLow or not smaHigh or not rsi or not dollar_volume_30 or not dollar_volume_90 or not market_cap:
            log.warning(f"stock price data missing for {symbol} - skipping")
            return
        
        if (beta > minBeta and beta < maxBeta and smaLow > smaHigh and rsi < rsiHigh and rsi > rsiLow):
            log.info(f"long: analyzing insider data")
            if dollar_volume_30 is None or dollar_volume_90 is None:
                log.info(f"long: {symbol} dollar volume change data not available - skipping")
                return
            
            if dollar_volume_30 > 0 and dollar_volume_30 > (0.9 * dollar_volume_90):
                log.info(f"long: analyzing earnings data")
                if market_cap*1000000 < minCap:
                    log.warning(f"market cap too low for {symbol} - skipping")
                    return
                
                earnings_score = 0

                if eps and eps_sma:
                    if eps > eps_sma:
                        earnings_score += 1
                if net_income and net_income_sma:
                    if net_income > net_income_sma:
                        earnings_score += 1
                if revenue and revenue_sma:
                    if revenue > revenue_sma:
                        earnings_score += 1
                if gross_margin and gross_margin_sma:
                    if gross_margin > gross_margin_sma:
                        earnings_score += 1
                if ebit and ebit_sma:
                    if ebit > ebit_sma:
                        earnings_score += 1
                if free_cash_flow and free_cash_flow_sma:
                    if free_cash_flow > free_cash_flow_sma:
                        earnings_score += 1
                if roe and roe_sma:
                    if roe > roe_sma:
                        earnings_score += 1
                
                log.debug(f"long: final earnings score for {symbol}: {earnings_score}")

                if earnings_score > 3:
                    log.debug(f"adding long symbol: {symbol}")
                    ensure_connection() # ensure mysql connection
                    LongStocks(
                        symbol=symbol,
                        beta=beta,
                        market_cap=market_cap,
                    ).save()
        return
    except Exception as err:
        log.error(f"error analyzing long symbol {symbol}: {err}", exc_info=True)
        return