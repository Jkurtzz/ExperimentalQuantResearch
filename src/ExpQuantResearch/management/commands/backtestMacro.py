import asyncio
import MySQLdb
import logging
import threading
import alpaca_trade_api
import numpy as np
import pandas_market_calendars as mcal
import time

import pandas as pd
import pytz
from core.config import config
from django.core.management.base import BaseCommand
from datetime import date, datetime, timedelta
from core.symbolUtils import get_short_term_market_status, get_long_term_market_status, get_med_term_market_status
from core.dbUtils import clear_db
from core.utils import plot_standardized_data   
from core.models import MonthlyMacroData, QuarterlyMacroData, AnnualMacroData
from core.stratTestUtils import get_stock_pct_change
log = logging.getLogger(__name__)

'''
to run test:
$python3 src/manage.py backtestMacro

macro selection current benchmark:
    - three models - short term (5 days), med term (10 days), long term (20 days)
    - short term 
        - does not have macro data
        - +- 0.75%
        - bull precision: 50%
        - bull accuracy: 35%
        - bear precision: 33%
        - bear accuracy: 5%
        - priority: last 
    - med term
        - has all data
        - +- 1%
        - bull precision: 51.7%
        - bull accuracy: 45%
        - bear precision: 100%
        - bear accuracy: 6%
        - priority: first
    - long term does not have news data
        - does not have news data
        - +- 3.0%
        - bull precision: 61%
        - bull accuracy: 38%
        - bear precision: n/a
        - bear accuracy: n/a
        - priority: second
    - final prediction stats - ie what trade we choose to do 
        - precision: 56.8%
        - accuracy: 50.7%
    - notes
        - bear meta feature of atr spike needed 
        - min precision: 0.72
        - med term model seems to be fine
            - short and long need massive improvements
    - next tests:
        1. increase min precision to 0.72, increase long term pct change to 3%, increase short term pct change to 0.75%
        2. remove random sampling
        3. lower long term window to 3 weeks
    - TODO:
        - add plot for precision and accuracy over time
        - consider looking into other indicators like RASI 
            - these show better interactions with market 
        - refactor code:
            - make market data one function instead of 3

    - NOTE:
        ideas and tests:
            - remove random sampling 
            - add interaction indicators 
            - get percent usage of indicators in correct and incorrect models
                - remove useless indicators
'''
class Command(BaseCommand):
    help = ''
    def handle(self, *args, **options):
        try:
            MonthlyMacroData.objects.all().delete()
            QuarterlyMacroData.objects.all().delete()
            AnnualMacroData.objects.all().delete()

            start_date = config.start_date
            end_date = datetime.today().date()
            # first get bullish/bearish/neutral results for each week
            # get spy data and indicators 
            url = 'https://data.alpaca.markets'
            api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

            spy_prices = {
                'date': [],
                'close': [],
            }
            response = api.get_bars(symbol='SPY', start=start_date, end=end_date, timeframe='1Day', limit=10000)
            log.debug(f"SPY data first index: {response[0].t} | final index: {response[-1].t}")
            for result in response:
                spy_prices['date'].append(result.t)
                spy_prices['close'].append(float(result.c))
            
            spy_df = pd.DataFrame(data=spy_prices)
            spy_df['date'] = pd.to_datetime(spy_df['date'])
            spy_df.set_index('date', inplace=True)

            test_start_date = datetime(year=2024, month=1, day=1, tzinfo=pytz.timezone('US/Eastern'))
            spy_df = spy_df[spy_df.index.weekday == 4]
            spy_df = spy_df[spy_df.index >= test_start_date]
            log.debug(f"spy df: {spy_df}")

            # get y-vars - if close increased x trading days later = 1, 0 otherwise
            # TODO: change timeframe to config var
            pos_pct_change_5d = [0] * len(spy_df['close'])
            neg_pct_change_5d = [0] * len(spy_df['close'])
            for i in range(len(spy_df['close']) - 1):
                pct_change = (spy_df['close'].iloc[i + 1] - spy_df['close'].iloc[i]) / np.abs(spy_df['close'].iloc[i])
                if pct_change > config.stocks.short_term_market.pos_pct_change:
                    pos_pct_change_5d[i] = 1

                if pct_change < config.stocks.short_term_market.neg_pct_change:
                    neg_pct_change_5d[i] = 1
            spy_df['pos_pct_change_5d'] = pd.Series(pos_pct_change_5d, index=spy_df.index)
            spy_df['neg_pct_change_5d'] = pd.Series(neg_pct_change_5d, index=spy_df.index)
            log.debug(f"short term benchmarks: total bull percentage: {float(sum(spy_df['pos_pct_change_5d']) / len(spy_df['close']))} | bear percentage: {float(sum(spy_df['neg_pct_change_5d']) / len(spy_df['close']))}")

            pos_pct_change_10d = [0] * len(spy_df['close'])
            neg_pct_change_10d = [0] * len(spy_df['close'])
            for i in range(len(spy_df['close']) - 2):
                pct_change = (spy_df['close'].iloc[i + 2] - spy_df['close'].iloc[i]) / np.abs(spy_df['close'].iloc[i])
                if pct_change > config.stocks.med_term_market.pos_pct_change:
                    pos_pct_change_10d[i] = 1

                if pct_change < config.stocks.med_term_market.neg_pct_change:
                    neg_pct_change_10d[i] = 1
            spy_df['pos_pct_change_10d'] = pd.Series(pos_pct_change_10d, index=spy_df.index)
            spy_df['neg_pct_change_10d'] = pd.Series(neg_pct_change_10d, index=spy_df.index)
            log.debug(f"med term benchmarks: total bull percentage: {float(sum(spy_df['pos_pct_change_10d']) / len(spy_df['close']))} | bear percentage: {float(sum(spy_df['neg_pct_change_10d']) / len(spy_df['close']))}")

            pos_pct_change_20d = [0] * len(spy_df['close'])
            neg_pct_change_20d = [0] * len(spy_df['close'])
            for i in range(len(spy_df['close']) - 4):
                pct_change = (spy_df['close'].iloc[i + 4] - spy_df['close'].iloc[i]) / np.abs(spy_df['close'].iloc[i])
                if pct_change > config.stocks.long_term_market.pos_pct_change:
                    pos_pct_change_20d[i] = 1

                if pct_change < config.stocks.long_term_market.neg_pct_change:
                    neg_pct_change_20d[i] = 1
            spy_df['pos_pct_change_20d'] = pd.Series(pos_pct_change_20d, index=spy_df.index)
            spy_df['neg_pct_change_20d'] = pd.Series(neg_pct_change_20d, index=spy_df.index)
            log.debug(f"long term benchmarks: total bull percentage: {float(sum(spy_df['pos_pct_change_20d']) / len(spy_df['close']))} | bear percentage: {float(sum(spy_df['neg_pct_change_20d']) / len(spy_df['close']))}")

            log.debug(f"results: {spy_df['neg_pct_change_5d']} | total 5-day bears: {sum(spy_df['neg_pct_change_5d'])}")
            log.debug(f"results: {spy_df['neg_pct_change_10d']} | total 10-day bears: {sum(spy_df['neg_pct_change_10d'])}")
            log.debug(f"results: {spy_df['neg_pct_change_20d']} | total 20-day bears: {sum(spy_df['neg_pct_change_20d'])}")

            # for each week, make prediction and compare to actual results
            total_weeks = 0

            total_short_term_bulls = 0
            total_short_term_bears = 0
            short_term_bull_success = 0
            short_term_bear_success = 0
            short_term_final_success = 0
            predicted_short_term_bulls = 0
            predicted_short_term_bears = 0
            correct_short_term_bulls = 0
            correct_short_term_bears = 0

            total_med_term_bulls = 0
            total_med_term_bears = 0
            med_term_bull_success = 0
            med_term_bear_success = 0
            med_term_final_success = 0
            predicted_med_term_bulls = 0
            predicted_med_term_bears = 0
            correct_med_term_bulls = 0
            correct_med_term_bears = 0
            
            total_long_term_bulls = 0
            total_long_term_bears = 0
            long_term_bull_success = 0
            long_term_bear_success = 0
            long_term_final_success = 0
            predicted_long_term_bulls = 0
            predicted_long_term_bears = 0
            correct_long_term_bulls = 0
            correct_long_term_bears = 0

            final_prediction_success = 0
            total_trades = 0
            total_correct_trades = 0

            cash = 25000
            current_trades = {}

            # market comparison values
            init_date = spy_df.index[0]

            # dataframe to plot progress vs SPY
            progress_dict = {
                'portfolio': [],
                'SPY': [],
                'date': [],
            }
            init_cash = cash

            for i in range(len(spy_df.index)):
                cur_datetime = spy_df.index[i]

                # update cash from trades
                log.debug(f"format check: cur date {cur_datetime.date()} | trade keys: {current_trades.keys()}")
                temp_cur_trades = {} # iterate through cur_trades, save non-expired trades to temp, then set cur_trades to temp to prevent loop errors
                for cur_trade_date in current_trades.keys():
                    if cur_datetime.date() >= cur_trade_date:
                        cur_profit_loss = current_trades[cur_trade_date]
                        log.debug(f"profit loss from trade that expired {cur_trade_date}: {cur_profit_loss}")
                        cash += cur_profit_loss
                    else:
                        temp_cur_trades[cur_trade_date] = current_trades[cur_trade_date]
                current_trades = temp_cur_trades

                log.debug(f"cash as of {cur_datetime}: {cash}")
                
                # compare to spy price
                url = 'https://data.alpaca.markets'
                api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

                response = api.get_bars(symbol='SPY', timeframe='1Day', feed='sip', sort='asc', start=init_date.replace(tzinfo=None).date(), end=cur_datetime.replace(tzinfo=None).date())
                init_spy_price = float(response[0].c)
                latest_spy_price = float(response[-1].c)
                num_shares = int(25000 / init_spy_price)
                profit_loss = (latest_spy_price - init_spy_price) * num_shares
                spy_cash = 25000 + profit_loss
                log.debug(f"spy cash on {cur_datetime}: {spy_cash}")
                time.sleep(0.2)

                # plot pct change progress vs market
                port_pct_change = (cash - init_cash) / init_cash
                spy_pct_change = (spy_cash - init_cash) / init_cash
                progress_dict['portfolio'].append(port_pct_change)
                progress_dict['SPY'].append(spy_pct_change)
                progress_dict['date'].append(cur_datetime)

                progress_df = pd.DataFrame(progress_dict)
                progress_df.set_index('date', inplace=True)
                plot_standardized_data(df=progress_df, title=f"current progress vs SPY on {cur_datetime.strftime('%Y-%m-%d')}")

                short_bull_res = spy_df['pos_pct_change_5d'][i]
                short_bear_res = spy_df['neg_pct_change_5d'][i]

                short_final_res = 0
                if short_bull_res == 1 and short_bear_res == 0:
                    short_final_res = 1
                if short_bull_res == 0 and short_bear_res == 1:
                    short_final_res = -1

                med_bull_res = spy_df['pos_pct_change_10d'][i]
                med_bear_res = spy_df['neg_pct_change_10d'][i]

                med_final_res = 0
                if med_bull_res == 1 and med_bear_res == 0:
                    med_final_res = 1
                if med_bull_res == 0 and med_bear_res == 1:
                    med_final_res = -1
                
                long_bull_res = spy_df['pos_pct_change_20d'][i]
                long_bear_res = spy_df['neg_pct_change_20d'][i]

                long_final_res = 0
                if long_bull_res == 1 and long_bear_res == 0:
                    long_final_res = 1
                if long_bull_res == 0 and long_bear_res == 1:
                    long_final_res = -1
                
                log.info(f"current date: {cur_datetime}")
                total_weeks += 1

                log.debug(f"making short term prediction")
                short_term_bull_prediction, short_term_bear_prediction, short_term_final_prediction = get_short_term_market_status(cutoff_date=cur_datetime)
                log.debug(f"making med term prediction")
                med_term_bull_prediction, med_term_bear_prediction, med_term_final_prediction = get_med_term_market_status(cutoff_date=cur_datetime)
                log.debug(f"making long term prediction")
                long_term_bull_prediction, long_term_bear_prediction, long_term_final_prediction = get_long_term_market_status(cutoff_date=cur_datetime)

                # get short term stats
                log.debug(f"{cur_datetime} short term bull prediction: {short_term_bull_prediction} | actual: {short_bull_res}")
                log.debug(f"{cur_datetime} short term bear prediction: {short_term_bear_prediction} | actual: {short_bear_res}")
                log.debug(f"{cur_datetime} short term final prediction: {short_term_final_prediction} | actual: {short_final_res}")

                if short_bull_res == 1:
                    total_short_term_bulls += 1
                if short_bear_res == 1:
                    total_short_term_bears += 1
                log.debug(f"{cur_datetime} total short term bulls: {total_short_term_bulls} | bears: {total_short_term_bears}")

                # accuracy 
                if short_term_bull_prediction == short_bull_res:
                    short_term_bull_success += 1
                if short_term_bear_prediction == short_bear_res:
                    short_term_bear_success += 1
                if short_term_final_prediction == short_final_res:
                    short_term_final_success += 1
                log.debug(f"{cur_datetime} short term accuracy stats: bull acc: {float(short_term_bull_success / total_weeks)} | bear acc: {float(short_term_bear_success / total_weeks)} | final acc: {float(short_term_final_success / total_weeks)}")

                # precision 
                if short_term_bull_prediction == 1:
                    predicted_short_term_bulls += 1
                    if short_bull_res == 1:
                        correct_short_term_bulls += 1
                    log.debug(f"{cur_datetime} short term bulls predicted: {predicted_short_term_bulls} | number of correct predictions: {correct_short_term_bulls} | precision: {float(correct_short_term_bulls / predicted_short_term_bulls)}")
                if short_term_bear_prediction == 1:
                    predicted_short_term_bears += 1
                    if short_bear_res == 1:
                        correct_short_term_bears += 1
                    log.debug(f"{cur_datetime} short term bears predicted: {predicted_short_term_bears} | number of correct predictions: {correct_short_term_bears} | precision: {float(correct_short_term_bears / predicted_short_term_bears)}")

                # get med term stats
                log.debug(f"{cur_datetime} med term bull prediction: {med_term_bull_prediction} | actual: {med_bull_res}")
                log.debug(f"{cur_datetime} med term bear prediction: {med_term_bear_prediction} | actual: {med_bear_res}")
                log.debug(f"{cur_datetime} med term final prediction: {med_term_final_prediction} | actual: {med_final_res}")

                if med_bull_res == 1:
                    total_med_term_bulls += 1
                if med_bear_res == 1:
                    total_med_term_bears += 1
                log.debug(f"{cur_datetime} total med term bulls: {total_med_term_bulls} | bears: {total_med_term_bears}")
                
                # accuracy 
                if med_term_bull_prediction == med_bull_res:
                    med_term_bull_success += 1
                if med_term_bear_prediction == med_bear_res:
                    med_term_bear_success += 1
                if med_term_final_prediction == med_final_res:
                    med_term_final_success += 1
                log.debug(f"{cur_datetime} med term accuracy stats: bull acc: {float(med_term_bull_success / total_weeks)} | bear acc: {float(med_term_bear_success / total_weeks)} | final acc: {float(med_term_final_success / total_weeks)}")

                # precision 
                if med_term_bull_prediction == 1:
                    predicted_med_term_bulls += 1
                    if med_bull_res == 1:
                        correct_med_term_bulls += 1
                    log.debug(f"{cur_datetime} med term bulls predicted: {predicted_med_term_bulls} | number of correct predictions: {correct_med_term_bulls} | precision: {float(correct_med_term_bulls / predicted_med_term_bulls)}")
                if med_term_bear_prediction == 1:
                    predicted_med_term_bears += 1
                    if med_bear_res == 1:
                        correct_med_term_bears += 1
                    log.debug(f"{cur_datetime} med term bears predicted: {predicted_med_term_bears} | number of correct predictions: {correct_med_term_bears} | precision: {float(correct_med_term_bears / predicted_med_term_bears)}")

                # get long term stats
                log.debug(f"{cur_datetime} long term bull prediction: {long_term_bull_prediction} | actual: {long_bull_res}")
                log.debug(f"{cur_datetime} long term bear prediction: {long_term_bear_prediction} | actual: {long_bear_res}")
                log.debug(f"{cur_datetime} long term final prediction: {long_term_final_prediction} | actual: {long_final_res}")

                if long_bull_res == 1:
                    total_long_term_bulls += 1
                if long_bear_res == 1:
                    total_long_term_bears += 1
                log.debug(f"{cur_datetime} total long term bulls: {total_long_term_bulls} | bears: {total_long_term_bears}")
                
                # accuracy 
                if long_term_bull_prediction == long_bull_res:
                    long_term_bull_success += 1
                if long_term_bear_prediction == long_bear_res:
                    long_term_bear_success += 1
                if long_term_final_prediction == long_final_res:
                    long_term_final_success += 1
                log.debug(f"{cur_datetime} long term accuracy stats: bull acc: {float(long_term_bull_success / total_weeks)} | bear acc: {float(long_term_bear_success / total_weeks)} | final acc: {float(long_term_final_success / total_weeks)}")

                # precision 
                if long_term_bull_prediction == 1:
                    predicted_long_term_bulls += 1
                    if long_bull_res == 1:
                        correct_long_term_bulls += 1
                    log.debug(f"{cur_datetime} long term bulls predicted: {predicted_long_term_bulls} | number of correct predictions: {correct_long_term_bulls} | precision: {float(correct_long_term_bulls / predicted_long_term_bulls)}")
                if long_term_bear_prediction == 1:
                    predicted_long_term_bears += 1
                    if long_bear_res == 1:
                        correct_long_term_bears += 1
                    log.debug(f"{cur_datetime} long term bears predicted: {predicted_long_term_bears} | number of correct predictions: {correct_long_term_bears} | precision: {float(correct_long_term_bears / predicted_long_term_bears)}")

                # total accuracy and precision
                if total_short_term_bulls + total_med_term_bulls + total_long_term_bulls > 0:
                    log.debug(f"{cur_datetime} current total bull accuracy: {float((correct_short_term_bulls + correct_med_term_bulls + correct_long_term_bulls) / (total_short_term_bulls + total_med_term_bulls + total_long_term_bulls))}")
                if predicted_short_term_bulls + predicted_med_term_bulls + predicted_long_term_bulls > 0:
                    log.debug(f"{cur_datetime} current total bull precision: {float((correct_short_term_bulls + correct_med_term_bulls + correct_long_term_bulls) / (predicted_short_term_bulls + predicted_med_term_bulls + predicted_long_term_bulls))}")
                if total_short_term_bears + total_med_term_bears + total_long_term_bears > 0:
                    log.debug(f"{cur_datetime} current total bear accuracy: {float((correct_short_term_bears + correct_med_term_bears + correct_long_term_bears) / (total_short_term_bears + total_med_term_bears + total_long_term_bears))}")
                if predicted_short_term_bears + predicted_med_term_bears + predicted_long_term_bears > 0:
                    log.debug(f"{cur_datetime} current total bear precision: {float((correct_short_term_bears + correct_med_term_bears + correct_long_term_bears) / (predicted_short_term_bears + predicted_med_term_bears + predicted_long_term_bears))}")

                # from predictions, determine timeframe and make trade
                make_trade = False 
                timeframe = 0
                trade = 'long'
                final_prediction = ''

                if med_term_final_prediction != 0:
                    total_trades += 1
                    make_trade = True 
                    timeframe = 14
                    if med_term_final_prediction == -1:
                        trade = 'short'

                    final_prediction = f"med_term_{trade}"
                    if med_term_final_prediction == med_final_res:
                        final_prediction_success += 1
                        total_correct_trades += 1
                elif long_term_final_prediction != 0:
                    total_trades += 1
                    make_trade = True 
                    timeframe = 28
                    if long_term_final_prediction == -1:
                        trade = 'short'

                    final_prediction = f"long_term_{trade}"
                    if long_term_final_prediction == long_final_res:
                        final_prediction_success += 1
                        total_correct_trades += 1
                elif short_term_final_prediction != 0:
                    total_trades += 1
                    make_trade = True
                    timeframe = 7 
                    if short_term_final_prediction == -1:
                        trade = 'short'

                    final_prediction = f"short_term_{trade}"
                    if short_term_final_prediction == short_final_res:
                        final_prediction_success += 1
                        total_correct_trades += 1
                else:
                    # final decision was to do nothing/predict neutral - check if none of the results are bulls
                    # success here equals correct short or avoid bear
                    if short_final_res != 1 and med_final_res != 1 and long_final_res != 1:
                        final_prediction_success += 1
                
                # if we made a correct prediction or dodged a bear 
                log.debug(f"{cur_datetime} final prediction accuracy: {float(final_prediction_success / total_weeks)}")
                # number of correct trades / total trades
                log.debug(f"{cur_datetime} final prediction precision: {float(total_correct_trades / total_trades)} ")

                if make_trade:
                    log.debug(f"{cur_datetime} final trade prediction: {final_prediction}")
                else:
                    log.debug(f"{cur_datetime} no strong prediction - skipping trade")

                # simulate trading
                if make_trade and len(current_trades.keys()) < 2:
                    trade_time = (cur_datetime + timedelta(days=3)).replace(hour=11, minute=30, second=0, microsecond=0) # monday
                    expire_time = trade_time + timedelta(days=timeframe) 

                    # check if market is open and switch to next open day if closed
                    nyse = mcal.get_calendar('NYSE')
                    
                    schedule = nyse.schedule(start_date=trade_time.replace(tzinfo=None).date(), end_date=trade_time.replace(tzinfo=None).date())
                    market_closed = schedule.empty
                    while market_closed:
                        trade_time = trade_time + timedelta(days=1)
                        schedule = nyse.schedule(start_date=trade_time.replace(tzinfo=None).date(), end_date=trade_time.replace(tzinfo=None).date())
                        market_closed = schedule.empty

                    schedule = nyse.schedule(start_date=expire_time.replace(tzinfo=None).date(), end_date=expire_time.replace(tzinfo=None).date())
                    market_closed = schedule.empty
                    while market_closed:
                        expire_time = expire_time + timedelta(days=1)
                        schedule = nyse.schedule(start_date=expire_time.replace(tzinfo=None).date(), end_date=expire_time.replace(tzinfo=None).date())
                        market_closed = schedule.empty

                    log.debug(f"format check: trade time: {trade_time} | expire time: {expire_time}")
                    init_price = float()
                    expire_price = float()
                    url = 'https://data.alpaca.markets'
                    api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

                    response = api.get_bars(symbol='SPY', timeframe='30Min', feed='sip', sort='asc', start=trade_time.replace(tzinfo=None).date(), end=expire_time.replace(tzinfo=None).date())
                    for result in response:
                        date = result.t.to_pydatetime()
                        if date == trade_time:
                            init_price = float(result.c)
                        if date == expire_time:
                            expire_price = float(result.c)

                    # make 'trade' and get profit/loss - able to hold 2x cash at a time - so we make a two-week trade every week - always holding 2x cash
                    num_shares = int((cash / 2) / init_price)
                    diff = expire_price - init_price
                    profit_loss = num_shares * diff

                    # if we shorted, reverse profit/loss
                    if trade == 'short':
                        profit_loss = -1 * profit_loss

                    log.debug(f"{final_prediction} done at {trade_time} | initial price: {init_price} | expire price: {expire_price} | profit/loss: {profit_loss}")
                    current_trades[expire_time.date()] = profit_loss
                
        except Exception as err:
            log.warning(f"error testing function: {err}", exc_info=True)


