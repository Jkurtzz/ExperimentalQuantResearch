import random
import threading
import logging
import time
import joblib
import pytz
import requests
import websockets
import alpaca_trade_api
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime, timedelta, time as dt_time
from django.utils import timezone
from core.config import config
from core.dbUtils import clear_db, ensure_connection
from core.symbolUtils import get_symbols, choose_stocks
from core.earningsUtils import get_earnings, get_earnings_from_db, realTimeQuarterlyReportData
from core.insiderUtils import get_insider_from_db, get_insider_transactions, realTimeInsiderData
from core.intraDayUtils import get_latest_price, get_boh_price, getLatestStockPriceFromDb, open_intraday_socket, intraday_real_time, getShortTermSma, dipOccurred, get_market_condition
from core.macroUtils import get_macro, get_macro_from_db, realTimeMacroData
from core.newsUtils import get_news_data, get_news_from_db
from core.pressUtils import get_press_from_db, get_press_releases, realTimePressData
from core.socialUtils import get_social_from_db, get_social_media_data, realTimeSocialData
from core.utils import plot_standardized_data
from core.models import AnnualMacroData, MonthlyMacroData, QEarnings, QuarterlyMacroData, TradeStats
log = logging.getLogger(__name__)

# global var for scheduled pre-market buys
premarket_trades = set()
multi_day_trades = set()

# dictionary to keep track of how many models predicts a trade
stock_trades_status = dict()

open_status = {'new', 'accepted', 'pending', 'open', 'partially_filled', 'held', 'pending_new', 'accepted_for_bidding'}
closed_status = {'filled', 'canceled', 'expired', 'rejected', 'closed', 'done_for_day', 'replaced', 'pending_cancel', 'pending_replace', 'stopped', 'suspended', 'calculated'}

# number of minutes before beginning of hour to wait for trade
lower_window = 10

# mutex lock for making orders
lock = threading.Lock()

# set of symbols blocked for trades for the day
blocked_stocks = set()

'''
End of every hour, get data
@param tradingPortfolio 
        key = stock symbol, val = {
            indicatortype: {
                    features: list
                    timeFrame: int
                }
            }, 
            ... 
'''
'''
create threads for each indicator type
for each thread
    get recent stock price - how do we do this with multiple threads?
    get recent indicator data
    make decision
'''
def start_real_time_calls(tradingPortfolio, tradingFrame):
    try:
        global stock_trades_status
        '''
        set up stock_trades_status dictionary - this keeps track of each symbol and how many models predict a price change
            - if 2 models predict a good trade, then we trade. otherwise, skip
        '''
        stock_trades_status = dict()

        # NOTE: do we need this?
        # threading.Thread(target=intraday_real_time, args=(symbols, datetime.today().date())).start()

        convertedPortfolio = convertTimeFrame(tradingPortfolio=tradingPortfolio)

        log.debug("starting realtime processes")

        if tradingFrame == 'daily':
            realTimeCallbackFxn = realTimeDailyDataAnalysis
        elif tradingFrame == 'intraday':
            realTimeCallbackFxn = realTimeDataAnalysis
        else:
            log.error("invalid trading type - cannot start realtime analysis")

        # TODO: split news into more threads or do fewer stocks
        insider_thread = threading.Thread(target=realTimeCallbackFxn, args=('insider', get_insider_transactions, convertedPortfolio['insider'], datetime.today().date() - timedelta(days=30)))
        press_thread = threading.Thread(target=realTimeCallbackFxn, args=('press', get_press_releases, convertedPortfolio['press'], datetime.today().date() - timedelta(days=10)))
        social_thread = threading.Thread(target=realTimeCallbackFxn, args=('social', get_social_media_data, convertedPortfolio['social'], datetime.today().date() - timedelta(days=1)))
        news_thread = threading.Thread(target=realTimeCallbackFxn, args=('news', get_news_data, convertedPortfolio['news'], datetime.today().date() - timedelta(days=1)))
        qearnings_thread = threading.Thread(target=realTimeCallbackFxn, args=('qearnings', get_earnings, convertedPortfolio['qearnings'], None))
        macro_thread = threading.Thread(target=realTimeCallbackFxn, args=('macro', get_macro, convertedPortfolio['macro'], None))
        
        insider_thread.start()
        press_thread.start()
        social_thread.start()
        news_thread.start()
        qearnings_thread.start()
        macro_thread.start()

        # threads will be done on friday at 5pm
        insider_thread.join()
        press_thread.join()
        social_thread.join()
        news_thread.join()
        qearnings_thread.join()
        macro_thread.join()

        return
    except Exception as err:
        log.error(f"error starting realtime calls: {err}", exc_info=True)

'''
gets most recent data, loads random forest model and determines if we should buy stock
@param tradingType - short or long
@param tradingFrame - daily or intraday
@param indicatorType - ex: news
@param symbol - stock symbol
@param features - features for random forests model
@param timeframe - hours in trade
@param pct_change - percent change model was trained on
'''

# TODO: update logs to better see data matching symbol and features, etc
def decide(tradingType, indicatorType, symbol, features, safety_features, tradingFrame, timeframe, pct_change):
    try:
        # define global vars 
        global premarket_trades
        global multi_day_trades
        global blocked_stocks
        global lock
        global stock_trades_status

        log.debug(f"determining trade for {symbol} - {tradingType} - {tradingFrame} - {indicatorType} at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))}")
        log.debug(f"current premarket stocks: {premarket_trades} | blocked stocks: {blocked_stocks} | multi-day trades: {multi_day_trades}")

        # get market condition - determine if shorts or longs are ideal for the next day
        market_condition = get_market_condition()
        if market_condition == tradingType:
            log.info(f"market conditions not ideal for {market_condition} - blocking trades for {symbol} | {tradingType} | {indicatorType}")
            return 

        # fxns for each indicator type to get most recent data
        callbacks = {
            "macro":  get_macro_from_db,  
            "qearnings": get_earnings_from_db,
            "insider": get_insider_from_db,
            "press": get_press_from_db,
            "news": get_news_from_db,
            "social": get_social_from_db,
        }

        # get data from db
        callbackFxn = callbacks[indicatorType]

        # macro uses different params then the others
        if indicatorType == "macro":
            _, df = callbackFxn(tradingFrame) # first response is static data, second is exp decay
        elif indicatorType == "qearnings": 
            _, df = callbackFxn(symbol, tradingFrame, False) # first response is raw data, second is exp decay
        elif indicatorType == 'insider':
            _, df = callbackFxn(symbol, tradingFrame) # first response is raw data, second is exp decay
        else:
            df = callbackFxn(symbol, tradingFrame)

        df_copy = df.copy() # needed for safety model

        # get features we need for this model
        df = df[features]

        # load ai model 
        model = joblib.load(f"models/trade/{tradingFrame}_{tradingType}_{symbol}_{indicatorType}.joblib")

        # get most recent data to predict
        entry = df.iloc[-1]
        log.debug(f"associated index with {symbol} - {tradingType} - {indicatorType} data: {df.index[-1]}")
        log.debug(f"{symbol} - {tradingType} most recent data: {entry}")
        log.debug(f"{tradingType} - {symbol} - {indicatorType} current time: {timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))}")

        # model expect 2d array of entry
        entry_arr = entry.values.reshape(1, -1)

        # make prediction
        prediction = model.predict(entry_arr)

        log.debug(f"{symbol} - {indicatorType} - {tradingType} prediction: {int(prediction)}")

        # safety prediction
        df_safety = df_copy[safety_features]

        safety_model = joblib.load(f"models/safety/{tradingFrame}_{tradingType}_{symbol}_{indicatorType}.joblib")

        safety_entry = df_safety.iloc[-1]
        log.debug(f"associated index with {symbol} - {tradingType} - {indicatorType} safety data: {df_safety.index[-1]}")
        log.debug(f"{symbol} - {tradingType} most recent safety data: {safety_entry}")
        log.debug(f"{tradingType} - {symbol} - {indicatorType} current time: {timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))}")

        # model expect 2d array of entry
        safety_entry_arr = safety_entry.values.reshape(1, -1)

        # make prediction
        safety_prediction = safety_model.predict(safety_entry_arr)

        # if safety model predicts opposite price movement; ie predicts price to increase when we are shorting, or predicts price to drop when we are longing - block trades for the day
        if int(safety_prediction) == 1:
            blocked_stocks.add(symbol)
        
        # if stock blocked, no trade
        if symbol in blocked_stocks:
            log.info(f"{symbol} blocked for day - no trades happening")
            return
        
        if int(prediction) == 1:
            log.info(f"{symbol} - {indicatorType} - {tradingFrame} predicted a {tradingType} - incrementing stock trades status dictionary")
            with lock:
                stock_trades_status[symbol] += 1

        log.debug(f"current stock trade status for {symbol}: {stock_trades_status[symbol]}")

        if int(stock_trades_status[symbol]) == 2:
            current_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))

            market_open = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
            market_open = market_open.replace(hour=9, minute=30, second=0, microsecond=0)

            # for daily trading - decision is made at 11pm night before
            if current_time > market_open:
                market_open = market_open + timedelta(days=1)

            # set market close to 10 minutes before close to give time to close any positions
            market_close = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
            market_close = market_close.replace(hour=15, minute=50, second=0, microsecond=0)

            # for daily trading - decision is made at 11pm night before
            if current_time > market_close:
                market_close = market_close + timedelta(days=1)

            if tradingFrame == 'intraday':
                expiration = df.index[-1] + timedelta(hours=timeframe)
            elif tradingFrame == 'daily': # if expiration is on the weekend, it gets set to monday in monitorStock
                expiration = df.index[-1] + timedelta(days=timeframe)
                expiration = expiration.replace(hour=11, minute=30, second=0, microsecond=0) # TEST: setting sell time to 11:30am of that day
            else:
                log.error("invalid tradingframe - cannot make trade")
                return
            # expiration = expiration.astimezone(pytz.timezone('UTC')).isoformat()

            log.debug(f"{symbol} - {tradingType} market open: {market_open} | market close: {market_close} | current time: {current_time} | exp: {expiration}")

            # premarket or if we are using daily strategy and plan a buy for the morning - daily trades should only hit the first if statement
            if tradingFrame == 'daily':
                if symbol not in multi_day_trades:
                    multi_day_trades.add(symbol)
                    log.debug(f"scheduled multi-day trades: {multi_day_trades}")
                    threading.Thread(target=multi_day_trade, args=(symbol, expiration, pct_change, tradingType)).start()
                else:
                    log.debug(f"multiday trade already scheduled for {symbol} - skipping")
            elif current_time < market_open:
                log.debug("premarket hours - scheduling a trade for 9:30")
                # add symbol to set of scheduled buys so we dont schedule to buy more than once
                if symbol not in premarket_trades:
                    premarket_trades.add(symbol)
                    log.debug(f"scheduled premarket trades: {premarket_trades}")
                    threading.Thread(target=pre_market_trade, args=(symbol, expiration, pct_change, tradingType)).start()
                else:
                    log.debug(f"{symbol} already scheduled for premarket trade - skipping")
            elif expiration <= market_close: # early open market
                log.debug("buying stock")
                threading.Thread(target=trade_stock, args=(symbol, expiration, pct_change, tradingType)).start()
            elif current_time > market_close:
                log.debug("market closed - not buying")
            else:
                log.debug("market soon to close - setting expiration to market close")
                threading.Thread(target=trade_stock, args=(symbol, market_close, pct_change, tradingType)).start()
        '''
            TODO: before making prediction
                - get current price and amount of money in account
        '''
    except Exception as err:
        log.error(f"unable to make buy decision: {err}", exc_info=True)

# multithreading execution
def runTask(func, args):
    return func(*args)

'''makes api call to alpaca to buy stock
    puts stop loss order in to prevent losses
    puts trail percentage in to sell upon increase
    @param symbol - stock symbol
    @param price - current price of stock. NOTE: may need to be retrieved at call time due to quickly changing prices
    @exp - experation - time when trade expires 
'''
def trade_stock(symbol, exp, pct_change, tradeType):
    try:
        global lock

        if tradeType == 'long':
            side = 'buy'
        elif tradeType == 'short':
            side = 'sell'
        else:
            log.error(f"{symbol} - {tradeType} error: invalid trade type - cannot buy/short stock")
            return

        url = config.alpaca.tradeUrl
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        # first determine if market is open
        is_open_res = api.get_clock()
        is_open = is_open_res.is_open
        if not is_open:
            log.warning(f"is open equals: {is_open}")
            log.warning(f"{symbol} - {tradeType} market closed at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))} - skipping trade")
            return

        latest_price = get_latest_price(symbol=symbol)
        

        log.debug(f"{symbol} - {tradeType} percent used for take profit: {1 + pct_change} | stpo loss: {(1 - ((2/3) * pct_change))}")
        '''
        pct_change = 1.5*ATR
        take profit = 1.5*ATR
        take profit = pct_change
        stop loss = 1 * ATR
        stop loss = 2/3 * pct_change
        '''
        take_profit = round(latest_price * (1 + pct_change), 2) 
        stop_loss = round(latest_price * (1 - ((2/3) * pct_change)), 2) 

        log.info(f"{symbol} - {tradeType} latest price: {latest_price} - stop loss: {stop_loss} - take profit: {take_profit}")
        
        # get cash we can use
        num_shares = int(25000 / latest_price)

        # mutex lock to prevent two buys at the same time
        with lock:
            # only buy if position not open for this symbol
            open_positions = {pos.symbol for pos in api.list_positions()}
            if symbol not in open_positions:

                # have to cancel any orders for this symbol - prevent other stop losses and take profits from interfering
                open_orders = api.list_orders(status="open")
                for order in open_orders:
                    if order.symbol == symbol:
                        log.debug(f"cancelling old order for {symbol}: {order.id}")
                        api.cancel_order(order.id)

                # if type is short - first check that we can trade the stock
                # NOTE: might not be needed because this is predetermined at stock selection
                if tradeType == 'short':
                    asset = api.get_asset(symbol=symbol)
                    if not asset.shortable:
                        log.warning(f"{symbol} not shortable - skipping trade")
                        return

                # trade stock
                current_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
                log.info(f"{side}ing {symbol} at {current_time}")
                order = api.submit_order(
                    symbol=symbol,
                    qty=num_shares,
                    side=side,
                    type='market',
                    time_in_force='gtc', 
                    order_class='bracket', 
                    take_profit={
                        'limit_price': take_profit
                    }, 
                    stop_loss={
                        'stop_price': stop_loss
                    },
                )

                # save data to db for stats
                # TODO: get indicator type for this
                TradeStats(
                    date=timezone.now().astimezone(tz=pytz.timezone('UTC')), 
                    order_id=order.id,
                    take_profit_id=order.legs[0].id,
                    stop_loss_id=order.legs[1].id,
                    pct_change=pct_change,
                    trade_type=tradeType,
                ).save()

                # monitor to see if time expires
                threading.Thread(target=monitor_stock, args=(symbol, num_shares, exp, order.legs[0].id, order.legs[1].id, tradeType)).start()
            else: 
                log.info(f"position open for {symbol} - skipping")

        return
    except Exception as err:
        log.error(f"error making trade: {err}", exc_info=True)


'''
sleeps until 9:30 and makes buy at open - meant as a thread
@params - same as buy_stock
'''
def pre_market_trade(symbol, exp, pct_change, tradeType):
    try:
        global premarket_trades
        log.debug(f"setting premarket {tradeType} for {symbol}")
        cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))

        market_open = cur_time.replace(hour=9, minute=30, second=0, microsecond=0)

        # should always happen - just incase
        if market_open > cur_time:
            log.debug(f"sleeping until {market_open} to make premarket {tradeType} for {symbol}")
            time.sleep((market_open - cur_time).total_seconds())

        # either buy or short
        trade_stock(symbol=symbol, exp=exp, pct_change=pct_change, tradeType=tradeType)
        
        # remove symbol from premarket_trades for next day
        premarket_trades.remove(symbol)
    except Exception as err:
        log.error(f"error scheduling premarket trade: {err}", exc_info=True)

# sleeps until ~11:30am to make trade - dodge early market volatility
def multi_day_trade(symbol, exp, pct_change, tradeType):
    try:
        global multi_day_trades

        log.debug(f"setting multi-day {tradeType} for {symbol}")
        cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))

        # multiday trades are done after volatile market calms down
        trade_time = cur_time.replace(hour=11, minute=30, second=0, microsecond=0)

        if cur_time > trade_time:
            log.info(f"current time is day before trade - adding day to trade time")
            trade_time = trade_time + timedelta(days=1)
        
        log.info(f"sleeping until {trade_time} to make multi-day {tradeType} for {symbol}")
        time.sleep((trade_time - cur_time).total_seconds())

        trade_stock(symbol=symbol, exp=exp, pct_change=pct_change, tradeType=tradeType)

        # remove symbol from multiday trades for next day
        multi_day_trades.remove(symbol)
    except Exception as err:
        log.error(f"error scheduling multiday trade for {symbol}: {err}", exc_info=True)

'''
waits until exp to sell position for symbol and qty of shares
'''
def monitor_stock(symbol, qty, exp, stopLossId, takeProfitId, tradingType):
    try:
        global open_status
        global closed_status

        if tradingType == 'long':
            side = 'sell'
        elif tradingType == 'short':
            side = 'buy' 
        else:
            log.error(f"{symbol} - {tradingType} error: invalid trading type - cannot monitor position")
            return

        cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))

        # for daily trades, if expiration is on weekend, set to monday
        # TODO: how to handle if market is closed on expiration?
        # TODO: how to handle if expiration goes over weekend but doesnt end on weekend? ie trade made on friday and should expire on wed but expires on monday
        if exp.weekday() == 5 or exp.weekday() == 6:
            exp = exp + timedelta(days=(7 - int(exp.weekday())))

        log.debug(f"sleeping til {exp} to liquidate {symbol} position")
        time.sleep((exp - cur_time).total_seconds())

        # sell position
        url = config.alpaca.tradeUrl
        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

        # check to see if position is still active - open_status is global list that defines strings that mean order is still open
        if api.get_order(stopLossId).status in open_status and api.get_order(takeProfitId).status in open_status:
            api.cancel_order(stopLossId)
            api.cancel_order(takeProfitId)
            log.info(f"{side}ing {symbol} position: {stopLossId} | {takeProfitId}")
            time.sleep(5)
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
        else:
            log.info(f"{symbol} position already closed - not liquidating")
    except Exception as err:
        log.error(f"error liquidating {symbol} position: {err}", exc_info=True)


'''
tradingPortfolio = {
    key = stock symbol, val = {
        indicatortype: {
                features: list
                timeFrame: int
                pct_change: float
            }
        }, 
        ... 
}


convert tradingPortfolio into a dictionary based by indicator type instead of by stock symbol

new dictionary:
indicator type = {
    key = symbol, value = {
        features: list(features),
        timeFrame: int,
        pct_change: float,
    }, ...
}
'''

def convertTimeFrame(tradingPortfolio):
    try:
        indicatorPortfolio = {
            'news': {}, 
            'press': {}, 
            'macro': {},
            'qearnings': {},
            'insider': {},
            'social': {},
        }
        for indicator in indicatorPortfolio.keys():
            for symbol in tradingPortfolio.keys():
                # if indicator type present in the symbol's portfolio, save it to new portfolio
                if indicator in tradingPortfolio[symbol].keys():    
                    indicatorPortfolio[indicator][symbol] = tradingPortfolio[symbol][indicator]
                    
        for indicator in indicatorPortfolio.keys():
            log.debug(f"portfolio for {indicator}")
            for key, vals in indicatorPortfolio[indicator].items():
                log.debug(f"key: {key} | vals: {vals}")
        
        return indicatorPortfolio
    except Exception as err:
        log.error(f"error converting portfolios: {err}", exc_info=True)


'''
1. loop and get data
2. at beginning of hour - make trade decision

params:
    @param indicatorType - the indicator type we are analysing here - ex: news
    @param indicatorCallBack - the function associated with the indicator type - ex: getNewsData for news
    @param tradingSet - dictionary with key = symbol, val = dict{
                                                            'features': list(features),
                                                            'timeFrame': int (hourly interval for trade)
                                                        }
    @param start_date - how early we want to look back for new data
'''
def realTimeDataAnalysis(indicatorType, indicatorCallBack, tradingSet, start_date):  
    try:
        global lower_window
        global blocked_stocks
        global premarket_trades

        symbols = list(tradingSet.keys())
        # randomize the order to prevent stocks from being traded at the end of the hour
        symbols = random.shuffle(symbols)
        log.debug(f"{indicatorType} randomized symbols: {symbols}")

        # enter loop - execute process every hour
        # every friday at 1600, leave loop and restart process (get new symbols)
        while timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).weekday() != 4 or timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).hour < 16:
            # refresh db connection
            ensure_connection()

            cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
            end_date = cur_time.date() + timedelta(days=1)
            next_hour = cur_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

            # sleep until beginning of hour - except social media due to delay
            # social media gets updates every hour at 14 minute mark, so get data then and then decide
            if indicatorType == 'social':
                next_hour = next_hour + timedelta(minutes=14)
                log.debug(f"social media data: sleeping until {next_hour}")
                time.sleep((next_hour - cur_time).total_seconds())

                for symbol in symbols:
                    # refresh db connection
                    ensure_connection()
                    indicatorCallBack(symbol, start_date, end_date, False)

                # make trade decision
                for symbol in symbols:
                    tradingType = tradingSet[symbol]['tradingType']
                    features = tradingSet[symbol]['features']
                    timeframe = tradingSet[symbol]['timeframe']
                    pct_change = tradingSet[symbol]['pct_change']
                    safety_features = tradingSet[symbol]['safety_features']
                    log.debug(f"making {tradingType} decision for {symbol} - {indicatorType} at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).time()} with {features} and timeframe: {timeframe}")
                    threading.Thread(target=decide, args=(tradingType, indicatorType, symbol, features, safety_features, 'intraday', timeframe, pct_change)).start()
            
            else: # for indicators that are not social media
                trading_lower_window = next_hour - timedelta(minutes=lower_window)

                # continuously get data until 10 minutes before beginning of our
                while (cur_time < trading_lower_window):
                    # get latest data for each symbol
                    for symbol in symbols:
                        # refresh db connection
                        ensure_connection()

                        log.debug(f"getting current {indicatorType} for {symbol} at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).time()}")

                        # get recent data and save to db - each indicator type takes different param sets
                        log.debug(f"getting recent {indicatorType} for {symbol}")
                        if indicatorType == 'macro':

                            # clear db data and retrieve new data - lazy but doesnt take long and is simple
                            MonthlyMacroData.objects.all().delete()
                            QuarterlyMacroData.objects.all().delete()
                            AnnualMacroData.objects.all().delete()

                            indicatorCallBack(config.start_date)
                        elif indicatorType == 'qearnings':

                            # clear entries and db and retrieve new ones - lazy but doesnt take much time
                            QEarnings.objects.filter(symbol=symbol).delete()

                            indicatorCallBack(symbol, config.start_date)
                        elif indicatorType == 'insider' or indicatorType == 'news':
                            indicatorCallBack(symbol, start_date, end_date)
                        else: # rest use the same params
                            indicatorCallBack(symbol, start_date, end_date, False)

                        cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
                        log.debug(f"time after getting {symbol} - {indicatorType} data: {cur_time}")

                    # update current time and do it again
                    cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))

                if cur_time > next_hour:
                    log.warning(f"current time greater than trading time - took too long to get data for {symbol} - {indicatorType}")
                    # increase the threshold so we dont miss trades
                    if lower_window < 20:
                        lower_window += 1
                    continue
                else:
                    # wait until beginning of hour - make decision then
                    cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
                    log.debug(f"current time: {cur_time} | sleeping until {next_hour}")
                    time.sleep((next_hour - cur_time).total_seconds())

                # if market closed - before 4 or after 4pm or on the weekend
                cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
                pre_market = cur_time.replace(hour=4, minute=0, second=0, microsecond=0)
                post_market = cur_time.replace(hour=16, minute=0, second=0, microsecond=0)
                if cur_time < pre_market or cur_time > post_market or cur_time.weekday() == 5 or cur_time.weekday() == 6:
                    log.debug(f"time: {cur_time} - market closed - skipping decision")

                    # if market closed, make sure blocked stocks and premarket buys are reset
                    blocked_stocks.clear()
                    premarket_trades.clear()
                    continue
                
                # make trade decision
                for symbol in symbols:
                    tradingType = tradingSet[symbol]['tradingType']
                    features = tradingSet[symbol]['features']
                    timeframe = tradingSet[symbol]['timeframe']
                    pct_change = tradingSet[symbol]['pct_change']
                    safety_features = tradingSet[symbol]['safety_features']
                    log.debug(f"making {tradingType} decision for {symbol} - {indicatorType} at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).time()} with {features} and timeframe: {timeframe}")
                    threading.Thread(target=decide, args=(tradingType, indicatorType, symbol, features, safety_features, 'intraday', timeframe, pct_change)).start()

        log.debug("end of trading week - exiting real time analysis")
    except Exception as err:
        log.error(f"error getting current {indicatorType} data: {err}", exc_info=True)


def realTimeDailyDataAnalysis(indicatorType, indicatorCallBack, tradingSet, start_date):  
    try:
        global lower_window
        global blocked_stocks
        global premarket_trades
        global multi_day_trades

        symbols = list(tradingSet.keys())
        # randomize the order to prevent stocks from being traded at the end of the hour
        symbols = random.shuffle(symbols)
        log.debug(f"{indicatorType} randomized symbols: {symbols}")

        # enter loop - execute process every hour
        # every friday at 1600, leave loop and restart process (get new symbols)
        # NOTE: this currently cant get hit - fix this
        while timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).weekday() != 4 or timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).hour < 16:
            # refresh db connection
            ensure_connection()

            # each day - reset stock trading status map
            for symbol in symbols:
                stock_trades_status[symbol] = 0

            cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
            end_date = cur_time.date() + timedelta(days=1)

            eod = cur_time.replace(hour=23, minute=0, second=0, microsecond=0)

            # if we already made decision for the night - add a day
            if cur_time > eod:
                eod = eod + timedelta(days=1)

            # sleep until 11pm to get data and make decision - except social media due to delay
            # social media gets updates every hour at 14 minute mark, so get data then and then decide
            if indicatorType == 'social':
                eod = eod + timedelta(minutes=14)
                log.debug(f"social media data: sleeping until {eod}")
                time.sleep((eod - cur_time).total_seconds())

                for symbol in symbols:
                    # refresh db connection
                    ensure_connection()

                    log.debug(f"getting current {indicatorType} for {symbol} at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).time()}")

                    indicatorCallBack(symbol, start_date, end_date, False)
                    # update current time
                    cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
                    log.debug(f"time after getting {symbol} - {indicatorType} data: {cur_time}")

                # if market closed - on friday or saturday night
                cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
                if cur_time.weekday() == 4 or cur_time.weekday() == 5:
                    log.debug(f"time: {cur_time} - market closed - skipping decision")
                    # if market closed, make sure we clear mult-day trading set
                    multi_day_trades.clear()
                    continue

                # make trade decision
                for symbol in symbols:
                    tradingType = tradingSet[symbol]['tradingType']
                    features = tradingSet[symbol]['features']
                    timeframe = tradingSet[symbol]['timeframe']
                    pct_change = tradingSet[symbol]['pct_change']
                    safety_features = tradingSet[symbol]['safety_features']
                    log.debug(f"making {tradingType} decision for {symbol} - {indicatorType} at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).time()} with {features} and timeframe: {timeframe}")
                    threading.Thread(target=decide, args=(tradingType, indicatorType, symbol, features, safety_features, 'daily', timeframe, pct_change)).start()
            
            else: # for indicators that are not social media

                # sleep until end of day to get data and make decision
                log.debug(f"{indicatorType} data: sleeping until {eod}")
                time.sleep((eod - cur_time).total_seconds())

                # get latest data for each symbol
                for symbol in symbols:
                    # refresh db connection
                    ensure_connection()

                    log.debug(f"getting current {indicatorType} for {symbol} at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).time()}")

                    # get recent data and save to db - each indicator type takes different param sets
                    if indicatorType == 'macro':

                        # clear db data and retrieve new data - lazy but doesnt take long and is simple
                        MonthlyMacroData.objects.all().delete()
                        QuarterlyMacroData.objects.all().delete()
                        AnnualMacroData.objects.all().delete()

                        indicatorCallBack(config.start_date)
                    elif indicatorType == 'qearnings':

                        # clear entries and db and retrieve new ones - lazy but doesnt take much time
                        QEarnings.objects.filter(symbol=symbol).delete()

                        indicatorCallBack(symbol, config.start_date)
                    elif indicatorType == 'insider' or indicatorType == 'news':
                        indicatorCallBack(symbol, start_date, end_date)
                    else: # rest use the same params - social and press use overwrite field to set start date based on last entry
                        indicatorCallBack(symbol, start_date, end_date, False)
                
                    # update current time
                    cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
                    log.debug(f"time after getting {symbol} - {indicatorType} data: {cur_time}")

                # if market closed - on friday or saturday night
                cur_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
                if cur_time.weekday() == 4 or cur_time.weekday() == 5:
                    log.debug(f"time: {cur_time} - market closed - skipping decision")
                    # if market closed, make sure we clear mult-day trading set
                    multi_day_trades.clear()
                    continue
                
                # make trade decision
                for symbol in symbols:
                    tradingType = tradingSet[symbol]['tradingType'] # short or long
                    features = tradingSet[symbol]['features']
                    timeframe = tradingSet[symbol]['timeframe']
                    pct_change = tradingSet[symbol]['pct_change']
                    safety_features = tradingSet[symbol]['safety_features']
                    log.debug(f"making {tradingType} decision for {symbol} - {indicatorType} at {timezone.now().astimezone(tz=pytz.timezone('US/Eastern')).time()} with {features} and timeframe: {timeframe}")
                    threading.Thread(target=decide, args=(tradingType, indicatorType, symbol, features, safety_features, 'daily', timeframe, pct_change)).start()

        log.debug("end of trading week - exiting real time analysis")
    except Exception as err:
        log.error(f"error getting current {indicatorType} data: {err}", exc_info=True)