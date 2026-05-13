import random
import threading
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from joblib import dump
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime, timedelta
from core.config import config
from core.dbUtils import clear_db
from core.symbolUtils import get_symbols, choose_stocks
from core.earningsUtils import get_earnings, get_earnings_from_db
from core.insiderUtils import get_insider_from_db, get_insider_transactions
from core.intraDayUtils import get_stock_price_data, get_stock_price_data_from_db
from core.macroUtils import get_macro, get_macro_from_db
from core.newsUtils import get_news_data, get_news_from_db
from core.pressUtils import get_press_from_db, get_press_releases
from core.socialUtils import get_social_from_db, get_social_media_data
from core.realTimeUtils import start_real_time_calls
from core.utils import plot_standardized_data, get_trade_stats
from core.trainingUtils import get_buy_instances, get_sell_instances, seq_train_model, get_best_feature_combination
log = logging.getLogger(__name__)

'''
1. gets list of stock symbols that fit criteria
2. chooses top x symbols to trade
3. for each symbol
    a. for each type of indicator (news, social, insider, press, q-reports, macro) 
            i. get historical data + save to db
            ii. create a thread that does:
                i. get data from db + clean data + calculate indicators
                ii. for each buy interval (1, 2, 3, 4, 5 hour) 
                    - train random forests models with indicators
                    - select model if accuracy and precision are high enough - not necessarily save 
                iii. save model with best results - 1 per indictor type
4. start realtime calls after
'''
def startup():
    try:
        # get stats of previous trades
        get_trade_stats()

        start_time = time.time()

        # clear database
        log.debug("clearing database")
        clear_db()

        # make api call to get df of symbols - save to db
        get_symbols(minCap=config.stocks.minCap, retry_attempts=2)

        log.debug(f"time to get symbols: {(time.time() - start_time) / 60} minutes")
        start_time = time.time()

        # pick top x stocks from db based on beta
        longSymbols, shortSymbols = choose_stocks(numStocks=config.stocks.numStocks)
        log.debug(f"long symbols chosen: {longSymbols}")
        log.debug(f"short symbols chosen: {shortSymbols}")

        log.debug(f"time to pick stocks: {(time.time() - start_time) / 60} minutes")
        start_time = time.time()

        # test symbols
        # longSymbols = ['PBA']
        # shortSymbols = ['FCN', 'ALNY', 'BSM', 'RNR', 'FLO', 'WTM', 'JNJ', 'PEP', 'UNM', 'EVRG', 'NSP', 'BAH', 'PCTY', 'CNA', 'D', 'CLX', 'MMS', 'LNT', 'HAS', 'GL', 'ETR', 'SOLV', 'ES', 'NEE']

        # get historical data for all symbols and save to db
        start_date = config.start_date
        end_date = datetime.today().date() + timedelta(days=1)

        # get long data
        for symbol in longSymbols:
            get_historical_data(symbol=symbol, start_date=start_date, end_date=end_date, tradingFrame=config.tradingFrame)
            log.debug(f"time to get historical data for {symbol}: {(time.time() - start_time) / 60} minutes")
            start_time = time.time()

        # get short data
        for symbol in shortSymbols:
            get_historical_data(symbol=symbol, start_date=start_date, end_date=end_date, tradingFrame=config.tradingFrame)
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
        for symbol in longSymbols:
            symbolDict = create_models(symbol=symbol, tradingType='long', tradingframe=tradingFrame)
            # if features returned, add to dict - otherwise, error occured
            if symbolDict != None:
                tradingPortfolio[symbol] = symbolDict
            log.debug(f"time to create ai model for {symbol}: {(time.time() - start_time) / 60} minutes")
            start_time = time.time()

        # retrieve stock data and train ai models
        for symbol in shortSymbols:
            symbolDict = create_models(symbol=symbol, tradingType='short', tradingframe=tradingFrame)
            # if features returned, add to dict - otherwise, error occured
            if symbolDict != None:
                tradingPortfolio[symbol] = symbolDict
            log.debug(f"time to create ai model for {symbol}: {(time.time() - start_time) / 60} minutes")
            start_time = time.time()
        
        # start real time process
        log.debug("trading portfolio")
        '''tradingPortfolio = '''
        log.debug(tradingPortfolio)

        # start_real_time_calls runs until 5pm on friday - so startup wont be called again until then
        start_real_time_calls(tradingPortfolio=tradingPortfolio, tradingFrame=tradingFrame)

        # restart process, get new symbols and retrain models
        log.debug("restarting process")
        # startup()
    except Exception as err:
        log.error(f"error starting app: {err}", exc_info=True)
        return
    

def get_historical_data(symbol, start_date, end_date, tradingFrame):
    try:
        news_thread = threading.Thread(target=get_news_data, args=(symbol, start_date, end_date))
        press_thread = threading.Thread(target=get_press_releases, args=(symbol, start_date, end_date, True))
        intra_thread = threading.Thread(target=get_stock_price_data, args=(symbol, start_date, end_date, tradingFrame, True))
        insider_thread = threading.Thread(target=get_insider_transactions, args=(symbol, start_date, end_date))
        social_thread = threading.Thread(target=get_social_media_data, args=(symbol, start_date, end_date, True))
        earnings_thread = threading.Thread(target=get_earnings, args=(symbol, start_date))

        news_thread.start()
        press_thread.start()
        intra_thread.start()
        insider_thread.start()
        social_thread.start()
        earnings_thread.start()

        news_thread.join()
        press_thread.join()
        intra_thread.join()
        insider_thread.join()
        social_thread.join()
        earnings_thread.join()
    except Exception as err:
        log.error(f"error getting historical data for {symbol}: {err}", exc_info=True)
        return
    
'''
    - get data from db
    - train many models to find best combination of features
    - save model as file

    @param symbol - stock symbol to get model for
    @param tradingType - short or long
    @param tradingFrame - daily or intraday
    @param end_date - cut off date for data - used for backtesting - only get data up til that date
    @return features - associated features upon completion

'''
def create_models(symbol, tradingType, tradingframe, end_date):
    try:

        log.info(f"creating indicator models for {symbol} - {tradingType} - {tradingframe} - {end_date}")

        ### use multithreading to test, train and save random forests models
        tasks = {
            "macro": (create_indicator_models, (symbol, get_macro_from_db, "macro", tradingType, tradingframe, end_date)),  
            "qearnings": (create_indicator_models, (symbol, get_earnings_from_db, "qearnings", tradingType, tradingframe, end_date)),
            "insider": (create_indicator_models, (symbol, get_insider_from_db, "insider", tradingType, tradingframe, end_date)),
            "press": (create_indicator_models, (symbol, get_press_from_db, "press", tradingType, tradingframe, end_date)),
            "news": (create_indicator_models, (symbol, get_news_from_db, "news", tradingType, tradingframe, end_date)),
            "social": (create_indicator_models, (symbol, get_social_from_db, "social", tradingType, tradingframe, end_date)),
        }

        results = {} 

        '''
            thread created for each type of indicator
            for each indicator:
                - get data from db, clean data and calculate indicators
                - test combinations of features and return best feature combination and timeframe 
                - return None if no models were strong enough
        '''
        with ThreadPoolExecutor() as executor:
            # submit different functions with their arguments
            futures = {
                executor.submit(func, *args): name
                for name, (func, args) in tasks.items()
            }
            
            # collect results as they complete
            for future in as_completed(futures):
                name = futures[future]  # get the variable name
                results[name] = future.result() # store data in associated 'task' key name

        # dictionary to keep track of features and timeframes of our models
        symbolDict = {}

        # if features and timeframe returns (ie, we have a succussful model), store and return dict with data
        log.debug(f"model results for {symbol} - {tradingType} - {tradingframe}")
        for key in results.keys():
            timeframe, features, pct_change, safety_features = results[key] # extract features and timeframe from results

            log.debug(f"current result: {key} | {timeframe} | {pct_change} | features: {features} | safety features: {safety_features}")
            if features != None: 
                symbolDict[key] = {
                    'tradingType': tradingType, 
                    'features': features,
                    'timeframe': timeframe,
                    'pct_change': pct_change,
                    'safety_features': safety_features,
                }

        # if no models were successful, return None
        if len(symbolDict.keys()) == 0:
            log.debug(f"no models saved for {tradingType} - {symbol}")
            return None
        
        log.debug(f"symbolDict for {tradingType} - {symbol}")
        log.debug(symbolDict)
        return symbolDict
    except Exception as err:
        log.error(f"error training model for {tradingType} - {symbol}: {err}", exc_info=True)
        return None
    


''' the following fxns get data from db, clean data and calculate indicators
    then trains ai model and returns best feature combination
    if error occurs or model not strong enough, return None 
@param get_data_from_db - function to get data from db for a specific type of indicator
@param indicatorType - name of type of indicator (news, insider, etc)
@param tradeType - short or long
@param tradingFrame - daily or intraday
@param end_date - cutoff date for data 
                - used for backtesting
                - datetime type
return features, timeframe
return None, None if error
'''
def create_indicator_models(symbol, get_data_from_db, indicatorType, tradeType, tradingFrame, end_date):
    try:
        log.debug(f"creating models for {symbol} - {indicatorType} - {tradeType}")

        ''' get callback function for getting buy/sell instances
        callback is the fxn used to get the trade signals
        safety callback gets the opposite trade signals - used for safety model to prevent invalid trades
        '''
        if tradeType == 'long':
            callbackFxn = get_buy_instances
            safetyCallbackFxn = get_sell_instances
        elif tradeType == 'short': 
            callbackFxn = get_sell_instances
            safetyCallbackFxn = get_buy_instances
        else:
            log.error(f"{symbol} - {indicatorType} - {tradeType} error: invalid trade type")
            return None, None, None, None

        # get intraday data and calculate buy instances
        df_prices = get_stock_price_data_from_db(symbol=symbol, tradingFrame=tradingFrame)
        # cut off data by end_date - used for backtesting
        df_prices = df_prices[df_prices.index <= end_date]

        # need latest atr for percent change
        pct_change = 1.5 * (df_prices['atr'][-1] / df_prices['close'][-1])

        # if shorting, reverse pct changes to be negative
        if tradeType == 'short': 
            pct_change = -1 * pct_change

        log.debug(f"{symbol} | {indicatorType} pct change: {pct_change}")

        df_prices = df_prices[['close']] # only columns needed here
        log.debug(f"{symbol} prices range: {df_prices.index}")

        # get data from db
        if indicatorType == "macro": # macro is the one instance that doesnt use a parameter
            _, df = get_data_from_db(tradingFrame) # first response is raw data, second is exp decay
        elif indicatorType == 'qearnings' :
            _, df = get_data_from_db(symbol, tradingFrame, False) # first response is raw data, second is exp decay
        elif indicatorType == 'insider':
            _, df = get_data_from_db(symbol, tradingFrame) # first response is raw data, second is exp decay
        else:
            df = get_data_from_db(symbol, tradingFrame) # callback function associated with specfic indicator type

        # check if data not available 
        if df is None:
            log.debug("empty dataframe - skipping")
            return None, None, None, None
        
        # cut of data by end date - used for backtesting
        df = df[df.index <= end_date]
        
        log.debug(f"{symbol} - {indicatorType} df range: {df.index}")

        num_feature = len(df.columns) # needed later when we create model
        log.debug(f"columns: {df.columns}")
        
        # lists of timeframes to test for making models - save model with best results
        match indicatorType:
            case "news":
                timeframes = config.finnhub.news.timeframes
            case "press":
                timeframes = config.finnhub.press.timeframes
            case "insider":
                timeframes = config.finnhub.insider.timeframes
            case "social":
                timeframes = config.finnhub.social.timeframes
            case "qearnings":
                timeframes = config.finnhub.earnings.timeframes
            case "macro":
                timeframes = config.finnhub.macro.timeframes
                
        log.debug(f"timeframes: {timeframes}")

        # get the best set of features, buy instances/ranges
        best_features = []
        best_accuracy = 0
        best_precision = 0
        best_timeframe = None
        best_pct_change = None

        ''' for each timeframe:
                for each percent change:
                    - get the buy instances and save as field 
                    - then get best feature combination
                    - if precisision and accuracy better than best, save features, timeframe and pct change
            '''
        for timeframe in timeframes:
            # create copies of dataframes for testing
            df_prices_copy = df_prices.copy()
            df_copy = df.copy()

            # get buy/short instances for respective timeframe and pct change
            df_prices_copy['signal'] = callbackFxn(df=df_prices_copy, timeframe=timeframe, pct_change=pct_change, tradingFrame=tradingFrame)

            # TEST: test only using data from 4am to 11am - only time we can buy/short
            if tradingFrame == 'intraday':
                df_prices_copy = df_prices_copy.between_time('04:00', '11:00')

            # get number of buys/shorts possible
            total_trades = 0
            for i in range(len(df_prices_copy['signal'])):
                if df_prices_copy['signal'][i] == 1:
                    total_trades += 1

            # if there are no instances of a price increase of %x in the next k hours, return
            log.debug(f"total {tradingFrame}-{tradeType}s for {symbol} at {pct_change}: {total_trades}")
            if total_trades == 0:
                log.info(f"no possible {tradeType}s for {symbol} - {indicatorType} - {pct_change} - skipping")
                continue

            # combine intraday with df for testing
            df_copy = df_copy.join(df_prices_copy[['signal']], how='inner')
            log.debug(f"joined {symbol} - {indicatorType} df index: {df_copy.index}")

            # get best feature combination
            features, precision, accuracy = get_best_feature_combination(df=df_copy, signal_col_name='signal', num_combs=750, min_comb_len=int(0.3 * num_feature), max_comb_len=int(0.8 * num_feature))
            if features is None or precision is None or accuracy is None:
                log.warning(f"error: training for {tradeType} - {symbol} - {timeframe} - {pct_change} returned None - skipping")
                continue

            # save if best precision and accuracy 
            if (precision + accuracy > best_precision + best_accuracy and precision >= config.models.minPrecision and accuracy >= config.models.minAccuracy and precision <= config.models.maxPrecision):
                best_precision = precision
                best_accuracy = accuracy
                best_features = features
                best_timeframe = timeframe
                best_pct_change = pct_change
            
        # if no strong model found - return None
        if best_timeframe == None:
            log.debug(f"no strong buy model found for {tradeType} - {symbol} - {indicatorType}")
            return None, None, None, None
        
        # create final models and save as file
        
        # create copies of dataframes for testing
        df_prices_copy = df_prices.copy()
        df_copy = df.copy()

        # get buy/short instances for respective timeframe and pct change
        df_prices_copy['signal'] = callbackFxn(df=df_prices_copy, timeframe=best_timeframe, pct_change=best_pct_change, tradingFrame=tradingFrame)

        # TEST: test only using data from 4am to 11am - only time we can buy
        if tradingFrame == 'intraday':
            df_prices_copy = df_prices_copy.between_time('04:00', '11:00')

        # combine intraday with df 
        df_copy = df_copy.join(df_prices_copy[['signal']], how='inner')

        # create model and save as file
        seq_train_model(df=df_copy, features=best_features, target_y='signal', fileName=f"trade/{tradingFrame}_{tradeType}_{symbol}_{indicatorType}")

        # log results 
        log.debug(f"best set of {tradeType} data for {indicatorType} for {symbol} date: {end_date} | precision: {best_precision} | accuracy: {best_accuracy} | best timeframe: {best_timeframe} - {tradingFrame} | best pct change: {best_pct_change} | features: {best_features}")
        
        # SAFETY MODEL CODE

        # if safety not enabled, return none, will be handled elsewhere
        best_safety_features = None 
        if config.models.safetyEnabled:
            log.debug(f"creating safety model for {symbol} - {indicatorType} - {tradeType}")

            # get features for safety model
            pct_change = (-3/4) * best_pct_change
            log.debug(f"{symbol} - {indicatorType} - {tradeType} pct change: {best_pct_change} | safety pct change: {pct_change}")

            # create copies of dataframes for testing
            df_prices_copy = df_prices.copy()
            df_copy = df.copy()

            # get buy/short instances for respective timeframe and pct change
            df_prices_copy['signal'] = safetyCallbackFxn(df=df_prices_copy, timeframe=best_timeframe, pct_change=pct_change, tradingFrame=tradingFrame)

            # TEST: test only using data from 4am to 11am - only time we can buy/short
            if tradingFrame == 'intraday':
                df_prices_copy = df_prices_copy.between_time('04:00', '11:00')

            # get number of buys/shorts possible
            total_trades = 0
            for i in range(len(df_prices_copy['signal'])):
                if df_prices_copy['signal'][i] == 1:
                    total_trades += 1

            # if there are no instances of a price increase of %x in the next k hours, return
            if total_trades == 0:
                log.info(f"no possible safety trades for {symbol} - {indicatorType} - {pct_change} - skipping")
                return None, None, None, None

            # combine intraday with df for testing
            df_copy = df_copy.join(df_prices_copy[['signal']], how='inner')
            log.debug(f"joined safety {symbol} - {indicatorType} df index: {df_copy.index}")

            # get best feature combination
            features, precision, accuracy = get_best_feature_combination(df=df_copy, signal_col_name='signal', num_combs=750, min_comb_len=int(0.3 * num_feature), max_comb_len=int(0.8 * num_feature))
            if features is None or precision is None or accuracy is None:
                log.warning(f"error: training for safety model {tradeType} - {symbol} - {timeframe} - {pct_change} returned None - skipping")
                return None, None, None, None

            # save if best precision and accuracy 
            if (precision >= config.models.minPrecision and accuracy >= config.models.minAccuracy and precision <= config.models.maxPrecision):
                best_safety_features = features
                # create model and save as file
                seq_train_model(df=df_copy, features=best_safety_features, target_y='signal', fileName=f"safety/{tradingFrame}_{tradeType}_{symbol}_{indicatorType}")
                log.debug(f"best set of safety {tradeType} data for {indicatorType} for {symbol} date: {end_date} | precision: {precision} | accuracy: {accuracy} | best timeframe: {best_timeframe} - {tradingFrame} | best pct change: {pct_change} | features: {best_safety_features}")
            else:
                log.warning(f"safety model for {symbol} - {indicatorType} - {tradeType} not strong enough - skipping")
                return None, None, None, None
        
        # END OF SAFETY MODEL CODE 
        return best_timeframe, best_features, best_pct_change, best_safety_features
    except Exception as err:
        log.error(f"error creating {tradeType} - {indicatorType} ai models for {symbol}: {err}", exc_info=True)
        return None, None, None, None