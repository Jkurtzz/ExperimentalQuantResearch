import random
import threading
import logging
import MySQLdb
import alpaca_trade_api
import time
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from joblib import dump
import pytz
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime, timedelta, date
from django.utils import timezone
from core.config import config
from core.dbUtils import clear_db
from core.symbolUtils import get_symbols, choose_stocks
from core.earningsUtils import get_earnings, get_earnings_from_db
from core.insiderUtils import get_insider_from_db, get_insider_transactions
from core.intraDayUtils import get_stock_price_data, get_stock_price_data_from_db, get_market_condition
from core.macroUtils import get_macro, get_macro_from_db
from core.newsUtils import get_news_data, get_news_from_db
from core.pressUtils import get_press_from_db, get_press_releases
from core.socialUtils import get_social_from_db, get_social_media_data
from core.realTimeUtils import start_real_time_calls, convertTimeFrame
from core.utils import plot_standardized_data, get_trade_stats
from core.control import get_historical_data, create_models
from core.models import BackTestTradeStats
log = logging.getLogger(__name__)

current_positions = dict() # k = symbol, v = dict: k = expiration, v = datetime | k = profit/loss, v = profit/loss
cash = float()

def test_strategy():
    try:
        # clear previous stats 
        BackTestTradeStats.objects.all().delete()

        global current_positions
        global cash

        cash = 25000 # starting cash amount

        init_cash = cash
        # dataframe to plot progress vs SPY
        progress_dict = {
            'portfolio': [],
            'SPY': [],
            'date': [],
        }

        # get series of datetimes
        log.info(f"getting base data to get series of tradeable days")
        start_date = config.start_date
        end_date = datetime.today().date()
        backtest_start_date = date(2024, 1, 1)
        timestamps = pd.date_range(start=backtest_start_date, end=end_date, freq='D', tz='US/Eastern')

        '''NOTE:
            Date info
                - dates is a series of tz-aware datetimes
                - when i need to use an individual date, i remove the tz and conver to date type
        '''
        dates = [dts.to_pydatetime() for dts in timestamps]

        log.info(f"date range for test: {dates}")

        # get first friday to start cycle
        i = 0
        cur_date = dates[0]
        while cur_date.weekday() != 4:
            i += 1
            cur_date = dates[i]
            log.debug(f"current day of week test: {cur_date.weekday()}")
            

        init_date = cur_date
        end_date = dates[-1]
        log.debug(f"start date: {cur_date} | final day: {end_date}")

        # begin cycle
        while cur_date < end_date:
            log.debug(f"current date: {cur_date} | current weekday: {cur_date.weekday()}")
            log.info(f"{cur_date}: current positions before cash check: {current_positions}")

            start_time = time.time()

            '''beginning of each weekday:
                - check current_positions dictionary
                - if any positions expire, record amount earned/lost
                - remove expired positions
            '''
            new_positions = dict() # will replace current positions at end
            for symbol, data in current_positions.items():
                expiration = data['expiration']
                profit_loss = float(data['profit_loss'])

                # if current date > expiration, record earnings, else add entry to new dictionary (instead of removing other entries to prevent index errors)
                if cur_date > expiration:
                    cash += profit_loss
                else:
                    new_positions[symbol] = data

            current_positions = new_positions
            log.info(f"{cur_date} cash: {cash}")

            # get pct change of SPY to compare to our strategy
            cash_pct_change = (cash - init_cash) / init_cash
            spy_pct_change = get_stock_pct_change(symbol='SPY', start_date=init_date.replace(tzinfo=None).date(), end_date=cur_date.replace(tzinfo=None).date())
            log.info(f"{cur_date} portfolio pct change: {cash_pct_change} | market pct change: {spy_pct_change}")

            progress_dict['portfolio'].append(cash_pct_change)
            progress_dict['SPY'].append(spy_pct_change)
            progress_dict['date'].append(cur_date)
            if cur_date.weekday() == 5: # plot results every week
                progress_df = pd.DataFrame(progress_dict, index=progress_dict['date'])
                progress_df = progress_df.drop(columns=['date'])
                plot_standardized_data(df=progress_df, title=f"current progress vs SPY on {cur_date.strftime('%Y-%m-%d')}")

            log.info(f"{cur_date} current positions after cash check: {current_positions}")

            # log our stats from the trades
            logBackTestStats(date=cur_date)

            ''' end of week - create new models
                in real application - done eod friday
                in testing - done on simulated saturday - easier this way and doesnt make a difference
            '''
            if cur_date.weekday() == 5: # simulated saturday
                log.info(f"creating new models on {cur_date}")
                # clear db before starting
                clear_db()

                # train new models 
                market_status = get_symbols(minCap=config.stocks.minCap, end_date=cur_date, retry_attempts=2)
                log.debug(f"time to get symbols: {(time.time() - start_time) / 60} minutes")
                start_time = time.time()

                # choose symbols 
                # TEST: hard code symbols for first loop to verify no bugs
                longSymbols, shortSymbols = choose_stocks(numStocks=config.stocks.numStocks)
                log.debug(f"long symbols chosen: {longSymbols}")
                log.debug(f"short symbols chosen: {shortSymbols}")
                          
                # get historical data - end date can be left at today - when models are created, end date will be set 
                # get long data
                for symbol in longSymbols:
                    get_historical_data(symbol=symbol, start_date=start_date, end_date=end_date.replace(tzinfo=None).date(), tradingFrame=config.tradingFrame)
                    log.debug(f"time to get historical data for {symbol}: {(time.time() - start_time) / 60} minutes")
                    start_time = time.time()

                # get short data
                for symbol in shortSymbols:
                    get_historical_data(symbol=symbol, start_date=start_date, end_date=end_date.replace(tzinfo=None).date(), tradingFrame=config.tradingFrame)
                    log.debug(f"time to get historical data for {symbol}: {(time.time() - start_time) / 60} minutes")
                    start_time = time.time()

                # get macro data (applies to all stocks)
                get_macro(start_date=start_date)

                '''key = stock symbol, val = {
                    [indicatortype: {
                            tradingType: str
                            features: list
                            timeframe: int
                            pct_change: float
                        safety_features: list
                        }
                    }, 
                    ... ]
                '''
                tradingPortfolio = {}
                tradingFrame = config.tradingFrame

                # retrieve stock data and train ai models
                long_prices = dict()
                for symbol in longSymbols:
                    symbolDict = create_models(symbol=symbol, tradingType='long', tradingframe=tradingFrame, end_date=cur_date)
                    # if features returned, add to dict - otherwise, error occured
                    if symbolDict != None:
                        tradingPortfolio[symbol] = symbolDict
                    log.debug(f"time to create ai model for {symbol}: {(time.time() - start_time) / 60} minutes")
                    start_time = time.time()

                    # get price data to plot results
                    url = 'https://data.alpaca.markets'
                    api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

                    long_prices[symbol] = []
                    long_prices['date'] = []
                    response = api.get_bars(symbol=symbol, timeframe='1Day', feed='sip', sort='asc', start=cur_date.replace(tzinfo=None).date(), end=(cur_date + timedelta(days=14)).replace(tzinfo=None).date())
                    init_price = float(response[0].c)
                    for result in response:
                        pct_change = (float(result.c) - init_price) / init_price
                        long_prices[symbol].append(pct_change)
                        long_prices['date'].append(result.t.to_pydatetime())

                dummy_long_df = pd.DataFrame(long_prices)
                dummy_long_df = dummy_long_df.set_axis(dummy_long_df['date'])
                plot_standardized_data(df=dummy_long_df, title=f"dummy long results as of {cur_date}")

                # retrieve stock data and train ai models
                short_prices = dict()
                for symbol in shortSymbols:
                    symbolDict = create_models(symbol=symbol, tradingType='short', tradingframe=tradingFrame, end_date=cur_date)
                    # if features returned, add to dict - otherwise, error occured
                    if symbolDict != None:
                        tradingPortfolio[symbol] = symbolDict
                    log.debug(f"time to create ai model for {symbol}: {(time.time() - start_time) / 60} minutes")
                    start_time = time.time()

                    # get price data to plot results
                    url = 'https://data.alpaca.markets'
                    api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

                    short_prices[symbol] = []
                    short_prices['date'] = []
                    response = api.get_bars(symbol=symbol, timeframe='1Day', feed='sip', sort='asc', start=cur_date.replace(tzinfo=None).date(), end=(cur_date + timedelta(days=14)).replace(tzinfo=None).date())
                    init_price = float(response[0].c)
                    for result in response:
                        pct_change = -1 * (float(result.c) - init_price) / init_price
                        short_prices[symbol].append(pct_change)
                        short_prices['date'].append(result.t.to_pydatetime())

                dummy_short_df = pd.DataFrame(short_prices)
                dummy_short_df = dummy_short_df.set_axis(dummy_short_df['date'])
                plot_standardized_data(df=dummy_short_df, title=f"dummy short results as of {cur_date}")

                log.info(f"test portfolio for {cur_date}: {tradingPortfolio}")


            # simulate trading decision for the day - do we do every loop?
            '''process:
                - for each day
                    - for each stock 
                        - predict the next day
                        - load up to current days data, make prediction
                        - if trade occurs, look ahead to see if stop loss or take profit is hit
                        - add/subtract from initial position
            '''
            # make predictions sunday night to thursday night - ie, if not friday or saturday, simulate prediction and trade day
            if cur_date.weekday() != 4 and cur_date.weekday() != 5:
                # first check if market was closed due to holiday
                nyse = mcal.get_calendar('NYSE')
                check_date = cur_date + timedelta(days=1) # checking if the day we are predicting (next day) is an open market day
                schedule = nyse.schedule(start_date=check_date.replace(tzinfo=None).date(), end_date=check_date.replace(tzinfo=None).date())
                is_open = not schedule.empty

                if is_open:
                    log.info(f"simulating trade day on {cur_date}")
                    start_time = time.time()
                    simulate_trading_day(tradingPortfolio=tradingPortfolio, cur_date=cur_date, market_status=market_status)
                    log.info(f"time it took to simulate {cur_date} trading: {(time.time() - start_time) / 60} minutes")
                    start_time = time.time()
                else: 
                    log.info(f"market closed on {cur_date} - skipping")
                
            i += 1
            cur_date = dates[i]
        return
    except Exception as err:
        log.error(f"error testing strategy: {err}", exc_info=True)
        return

'''
@param tradingPortfolio - dict
@param cur_date - datetime
@param market_status - string 
'''
def simulate_trading_day(tradingPortfolio, cur_date, market_status):
    try:
        global current_positions
        global cash         

        # fxns for each indicator type to get most recent data
        callbacks = {
            "macro":  get_macro_from_db,  
            "qearnings": get_earnings_from_db,
            "insider": get_insider_from_db,
            "press": get_press_from_db,
            "news": get_news_from_db,
            "social": get_social_from_db,
        }

        blocked_stocks = set()
        stock_trades_status = dict() # k = symbol, v = int - if int == 2, make trade      
        activeSymbols = tradingPortfolio.keys()
        for symbol in activeSymbols:
            stock_trades_status[symbol] = 0

        # iterate through each indicatortype-stock combo and make prediction
        convertedPortfolio = convertTimeFrame(tradingPortfolio=tradingPortfolio)

        # get market mtm for the day - determines what trades we are blocking for the day
        market_momentum = get_market_condition(end_date=cur_date.replace(tzinfo=None).date())
        log.info(f"market momentum on {cur_date.replace(tzinfo=None).date()}: {market_momentum} blocked")

        scheduled_trades = dict() # k = symbol, val = dict{tradingType, timeframe, pct_change, indicatorTypes=list}
        
        # make predictions
        # for each indicator type and associated data
        for indicatorType, indicatorTypeTradingSet in convertedPortfolio.items():
            # for each symbol and modelInfo - the features, timeframe, pct change, and safety features associated with this model
            for symbol, modelInfo in indicatorTypeTradingSet.items():
                tradingType = modelInfo['tradingType']
                features = modelInfo['features']
                timeframe = modelInfo['timeframe']
                pct_change = modelInfo['pct_change']
                safety_features = modelInfo['safety_features']

                log.info(f"making trade decision for {symbol} - {indicatorType} - {tradingType} on {cur_date}")

                # if blocked market status equals our trading type (short or long), skip decision
                if market_momentum == tradingType:
                    log.info(f"test simulation {cur_date}: market status is {market_momentum} - skipping decision for {symbol} - {indicatorType} - {tradingType}")
                    continue

                # get data and cut off at cur_date
                callbackFxn = callbacks[indicatorType]

                # macro uses different params then the others
                if indicatorType == "macro":
                    _, df = callbackFxn('daily') # first response is static data, second is exp decay
                elif indicatorType == "qearnings": 
                    _, df = callbackFxn(symbol, 'daily', False) # first response is raw data, second is exp decay
                elif indicatorType == 'insider':
                    _, df = callbackFxn(symbol, 'daily') # first response is raw data, second is exp decay
                else:
                    df = callbackFxn(symbol, 'daily')

                df = df[df.index <= cur_date]

                df_copy = df.copy() # needed for safety model

                # get features we need for this model
                df = df[features]
                
                # load ai model 
                model = joblib.load(f"models/trade/daily_{tradingType}_{symbol}_{indicatorType}.joblib")

                # get most recent data to predict
                entry = df.iloc[-1]
                log.debug(f"test simulation {cur_date}: associated index with {symbol} - {tradingType} - {indicatorType} data: {df.index[-1]}")
                log.debug(f"test simulation {cur_date}: {symbol} - {tradingType} most recent data: {entry}")

                # model expect 2d array of entry
                entry_arr = entry.values.reshape(1, -1)

                # make prediction
                prediction = model.predict(entry_arr)

                log.debug(f"test simulation {cur_date}: {symbol} - {indicatorType} - {tradingType} prediction: {int(prediction)}")

                # safety prediction
                if config.models.safetyEnabled:
                    df_safety = df_copy[safety_features]

                    safety_model = joblib.load(f"models/safety/daily_{tradingType}_{symbol}_{indicatorType}.joblib")

                    safety_entry = df_safety.iloc[-1]
                    log.debug(f"test simulation {cur_date}: associated index with {symbol} - {tradingType} - {indicatorType} safety data: {df_safety.index[-1]}")
                    log.debug(f"test simulation {cur_date}: {symbol} - {tradingType} most recent safety data: {safety_entry}")

                    # model expect 2d array of entry
                    safety_entry_arr = safety_entry.values.reshape(1, -1)

                    # make prediction
                    safety_prediction = safety_model.predict(safety_entry_arr)

                    # if safety model predicts opposite price movement; ie predicts price to increase when we are shorting, or predicts price to drop when we are longing - block trades for the day
                    if int(safety_prediction) == 1:
                        blocked_stocks.add(symbol)
                
                    # if stock blocked, no trade
                    if symbol in blocked_stocks:
                        log.info(f"test simulation {cur_date}: {symbol} blocked for day - no trades happening")
                        continue
                
                if int(prediction) == 1:
                    # TEST: if ensemble enabled and market status is not bullish - let it rip in bullish market
                    # if config.models.ensembleEnabled:
                    #     log.info(f"test simulation {cur_date}: {symbol} - {indicatorType} - daily predicted a {tradingType} - incrementing stock trades status dictionary")
                    #     # NOTE: need to still save indicator type to dictionary
                    #     stock_trades_status[symbol] += 1
                    #     log.debug(f"test simulation {cur_date}: current stock trade status for {symbol}: {stock_trades_status[symbol]}")
                    #     if stock_trades_status[symbol] < 2:
                    #         continue
                    
                    if symbol in scheduled_trades.keys():
                        indicatorTypes = scheduled_trades[symbol]['indicatorTypes']
                        indicatorTypes.append(indicatorType)
                        # save the longest timeframe
                        if scheduled_trades[symbol]['timeframe'] > timeframe:
                            continue

                        scheduled_trades[symbol] = {
                            'timeframe': timeframe,
                            'tradingType': tradingType, # shouldnt be different
                            'pct_change': pct_change, # shouldnt be different
                            'indicatorTypes': indicatorTypes
                        } 
                    else:
                        scheduled_trades[symbol] = {
                            'timeframe': timeframe,
                            'tradingType': tradingType,
                            'pct_change': pct_change,
                            'indicatorTypes': [indicatorType]
                        }
                    log.debug(f"test simulation scheduled trades: {scheduled_trades}")
        
        # get position sizes and schedule trades
        for symbol, tradeData in scheduled_trades.items():
            timeframe = tradeData['timeframe']
            tradingType = tradeData['tradingType']
            pct_change = tradeData['pct_change']
            indicatorTypes = tradeData['indicatorTypes']

            if config.models.ensembleEnabled:
                if len(indicatorTypes) < 2:
                    log.info(f"test simulation: not enough predictions for {symbol} - skipping")
                    continue

            # check if cash available 
            # NOTE: update this code when we switch to ATR-based position sizing
            # NOTE: update this to split position size based on cash / num of schuled trades
            position_size = 5000 # update later
            num_cur_positions = len(current_positions.keys())
            max_possible_positions = int((2*cash) / position_size)
            log.info(f"number of current positions: {num_cur_positions} | max possible positions: {max_possible_positions}")
            if num_cur_positions >= max_possible_positions:
                log.info(f"no cash available to trade - skipping {symbol} - {indicatorType} trade")
                continue

            # if position already open for this stock, skip
            if symbol in current_positions.keys():
                log.info(f"position already open for {symbol} - skipping")
                continue

            # simulate trade - start date set to next day, thats when trade would happen. expiration will be timeframe + next day
            simulate_trade(symbol=symbol, start_date=(cur_date + timedelta(days=1)), timeframe=timeframe, pct_change=pct_change, tradeType=tradingType, indicatorTypes=indicatorTypes, position_size=position_size)

        return
    except Exception as err:
        log.error(f"error simulating trading day: {err}", exc_info=True)
        return

''' 
@param symbol - str
@param start_date - datetime
@param timeframe - int
@param pct_change - float
@param tradeType - str
@param indicatorTypes - list
@param position_size - float
'''
def simulate_trade(symbol, start_date, timeframe, pct_change, tradeType, indicatorTypes, position_size):
    try:
        log.info(f"simulating {tradeType} for {symbol}")
        global current_positions
        global cash
        init_price = float()
        stop_loss = float()
        take_profit = float()
        num_shares = int()

        # get price at 11:30
        expected_date = start_date.replace(hour=11, minute=30, second=0, microsecond=0)
        expiration = expected_date + timedelta(days=timeframe)

        # incase expiration lands on weekend
        if expiration.weekday() == 5 or expiration.weekday() == 6:
            expiration = expiration + timedelta(days=(7 - int(expiration.weekday())))

        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        response = api.get_bars(symbol=symbol, timeframe='30Min', feed='sip', sort='asc', start=start_date.replace(tzinfo=None).date(), end=start_date.replace(tzinfo=None).date())
        for result in response:
            date = result.t.to_pydatetime()
            log.info(f"price time as eastern: {date} | expected date: {expected_date}")

            if date == expected_date:
                init_price = result.c
                num_shares = int(position_size / init_price)
                log.info(f"initial price for {symbol} on {start_date}: {init_price} | number of shares obtained: {num_shares}")
                '''
                pct_change = 1.5*ATR
                take profit = 1.5*ATR
                take profit = pct_change
                stop loss = 1 * ATR
                stop loss = 2/3 * pct_change
                '''
                take_profit = round(init_price * (1 + pct_change), 2)
                stop_loss = round(init_price * (1 - ((2/3) * pct_change)), 2)
                log.info(f"{symbol} - {tradeType} initial price: {init_price} | stop loss: {stop_loss} | take profit: {take_profit}")
                break
        
        if not init_price:
            log.error(f"could not get stock price at 11:30 eastern for {symbol} - skipping trade")
            return
        
        # traverse prices moving forward to find stop loss or take profit
        # NOTE: needs to be after trade time - ie we are getting data from midnight of the day of the trade
        response = api.get_bars(symbol=symbol, timeframe='15Min', feed='sip', sort='asc', start=expected_date.replace(tzinfo=None).date(), end=expiration.replace(tzinfo=None).date(), limit=10000)

        # create df to plot trade chart
        prices = {
            'date': [],
            'price': []
        }
        for i in response:
            prices['date'].append(i.t.to_pydatetime())
            prices['price'].append(float(i.c))

        price_df = pd.DataFrame(prices)
        price_df = price_df.set_index('date')
        plot_standardized_data(df=price_df, title=f"{symbol} - {indicatorTypes} - {tradeType} trade | trade date: {expected_date.replace(tzinfo=None).date()} | exp: {expiration.replace(tzinfo=None).date()} | init price: {init_price} | stop loss: {stop_loss} | take profit: {take_profit}")

        if tradeType == 'long':
            for result in response:
                # unable to remove extended hours, so have to check that as well
                date = result.t.to_pydatetime()

                # NOTE: temporary fix
                if date < expected_date:
                    log.debug(f"date is before trade time: {date} - trade date: {expected_date} - skipping")
                    continue

                market_open = date.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = date.replace(hour=16, minute=0, second=0, microsecond=0)
                log.info(f"long: date: {date} | market_open: {market_open} | market_close: {market_close} | expiration: {expiration}")
                log.info(f"long: date: {date} | cur price: {float(result.c)} | stop loss: {stop_loss} | take profit: {take_profit}")

                if date < market_open or date > market_close:
                    log.info("long: market not open - skipping")
                    continue

                # should be checking open, close, high, and low - but close should be sufficient
                close = float(result.c)
                if date >= expiration:
                    # get profit/loss and save results
                    diff = close - init_price
                    profit_loss = num_shares * diff
                    log.info(f"long: expiration hit at {date} - liquidating position | price: {result.c} | profit/loss: {profit_loss}")
                    current_positions[symbol] = {
                        'expiration': date,
                        'profit_loss': profit_loss,
                    }
                    break
                elif close > take_profit:
                    # get profit and save results
                    diff = close - init_price
                    profit = num_shares * diff
                    log.info(f"long: take profit hit at {date} | price: {result.c} | take profit: {take_profit} | profit: {profit}")
                    current_positions[symbol] = {
                        'expiration': date,
                        'profit_loss': profit,
                    }

                    # save stats to db
                    for indicatorType in indicatorTypes:
                        BackTestTradeStats(
                            date=date,
                            symbol=symbol,
                            success=True,
                            indicator_type=indicatorType, 
                            pct_change=pct_change,
                            trade_type=tradeType,
                        ).save()
                    break
                elif close < stop_loss:
                    # get loss and save results
                    diff = close - init_price
                    loss = num_shares * diff
                    log.info(f"long: stop loss hit at {date} | price: {result.c} | stop loss: {stop_loss} | profit: {loss}")
                    current_positions[symbol] = {
                        'expiration': date,
                        'profit_loss': loss,
                    }
                    
                    # save stats to db
                    for indicatorType in indicatorTypes:
                        BackTestTradeStats(
                            date=date,
                            symbol=symbol,
                            success=False,
                            indicator_type=indicatorType, 
                            pct_change=pct_change,
                            trade_type=tradeType,
                        ).save()
                    break
        # handle short
        else: 
            for result in response:
                # unable to remove extended hours, so have to check that as well
                date = result.t.to_pydatetime()
                market_open = date.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = date.replace(hour=16, minute=0, second=0, microsecond=0)
                log.info(f"short: date: {date} | market_open: {market_open} | market_close: {market_close} | expiration: {expiration}")
                log.info(f"short: date: {date} | cur price: {float(result.c)} | stop loss: {stop_loss} | take profit: {take_profit}")

                if date < market_open or date > market_close:
                    log.info("short: market not open - skipping")
                    continue

                # should be checking open, close, high, and low - but close should be sufficient
                close = float(result.c)
                if date >= expiration:
                    # get profit/loss and save results
                    diff = close - init_price
                    profit_loss = -1 * (num_shares * diff)
                    log.info(f"short: expiration hit at {date} - liquidating position | price: {result.c} | profit/loss: {profit_loss}")
                    current_positions[symbol] = {
                        'expiration': date,
                        'profit_loss': profit_loss,
                    }
                    break
                elif close < take_profit:
                    # get profit and save results
                    diff = close - init_price
                    profit = -1 * (num_shares * diff)
                    log.info(f"short: take profit hit at {date} | price: {result.c} | take profit: {take_profit} | profit: {profit}")
                    current_positions[symbol] = {
                        'expiration': date,
                        'profit_loss': profit,
                    }

                    # save stats to db
                    for indicatorType in indicatorTypes:
                        BackTestTradeStats(
                            date=date,
                            symbol=symbol,
                            success=True,
                            indicator_type=indicatorType, 
                            pct_change=pct_change,
                            trade_type=tradeType,
                        ).save()
                    break
                elif close > stop_loss:
                    # get loss and save results
                    diff = close - init_price
                    loss = -1 * (num_shares * diff)
                    log.info(f"short: stop loss hit at {date} | price: {result.c} | stop loss: {stop_loss} | profit: {loss}")
                    current_positions[symbol] = {
                        'expiration': date,
                        'profit_loss': loss,
                    }

                    # save stats to db
                    for indicatorType in indicatorTypes:
                        BackTestTradeStats(
                            date=date,
                            symbol=symbol,
                            success=False,
                            indicator_type=indicatorType, 
                            pct_change=pct_change,
                            trade_type=tradeType,
                        ).save()
                    break
        return
    except Exception as err:
        log.error(f"error simulating trade: {err}", exc_info=True)
        return

'''
    @param symbol
    @param start_date, end_date - date type
'''
def get_stock_pct_change(symbol, start_date, end_date):
    try:
        if start_date == end_date:
            return None
        
        url = 'https://data.alpaca.markets'
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        response = api.get_bars(symbol=symbol, timeframe='1Day', feed='sip', sort='asc', start=start_date, end=end_date, limit=10000)

        init = response[0]
        init_close = float(init.c)
        log.debug(f"init date and price for {symbol}: {init.t} - {init_close}")

        latest = response[-1]
        latest_close = float(latest.c)
        log.debug(f"latest date and price for {symbol}: {latest.t} - {latest_close}")

        pct_change = (latest_close - init_close) / init_close
        log.info(f"pct change for {symbol}: {pct_change}")
        return pct_change
    except Exception as err:
        log.error(f"error getting pct change for {symbol}: {err}", exc_info=True)
        return None
    

def logBackTestStats(date):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_backtesttradestats;"
        df = pd.read_sql(query, conn)

        if len(df.index) == 0:
            return
        
        log.debug(f"Trade Stats as of {date}")

        success_series = df['success']
        num_success = (success_series == 1).sum()
        log.debug(f"total trades: {len(df['success'])} | successful trades: {num_success} | success rate: {float(num_success / len(success_series))}")

        # stats for longs
        long_df = df[df['trade_type'] == 'long']
        long_success_series = long_df['success']
        num_success_longs = (long_success_series == 1).sum()
        log.debug(f"total longs: {len(long_success_series)} | successful longs: {num_success_longs} | success rate: {float(num_success_longs / len(long_success_series))}")

        # stats for shorts
        short_df = df[df['trade_type'] == 'short']
        short_success_series = short_df['success']
        num_success_shorts = (short_success_series == 1).sum()
        log.debug(f"total shorts: {len(short_success_series)} | successful shorts: {num_success_shorts} | success rate: {float(num_success_shorts / len(short_success_series))}")

        conn.close()
        return
    except Exception as err:
        log.error(f"error logging backtest trades stats: {err}", exc_info=True)
        return