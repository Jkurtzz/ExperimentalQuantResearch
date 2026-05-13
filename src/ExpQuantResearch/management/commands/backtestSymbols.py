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
from core.symbolUtils import get_market_status
from core.dbUtils import clear_db
from core.utils import plot_standardized_data   
from core.models import MonthlyMacroData, QuarterlyMacroData, AnnualMacroData
from core.stratTestUtils import get_stock_pct_change
log = logging.getLogger(__name__)

'''
to run test:
$python3 src/manage.py backtestSymbols
'''
class Command(BaseCommand):
    help = ''
    def handle(self, *args, **options):
        try:
            '''process:
                    - get the market status for the next 10 trading days 
                    - launch strategy for returned market status (test only bullish for now)
                    - get selected symbols
                    - long/short only those symbols
                    - see if outperforms the market after 10 days
                    - repeat
            '''

            # get market status 
            start_date = config.start_date
            end_date = datetime.today().date()

            url = 'https://data.alpaca.markets'
            api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')
            spy_prices = {
                'date': [],
                'close': [],
            }

            response = api.get_bars(symbol='SPY', start=start_date, end=end_date, timeframe='1Day')
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

            # get y-vars - if close increased/decreases/stayed over the next 10 trading days: 1, 0, -1
            pct_change_10d = [0] * len(spy_df['close'])
            for i in range(len(spy_df['close']) - 2):
                pct_change = (spy_df['close'].iloc[i + 2] - spy_df['close'].iloc[i]) / np.abs(spy_df['close'].iloc[i])
                if pct_change > config.stocks.market.pct_change:
                    pct_change_10d[i] = 1
                elif pct_change < -1 * config.stocks.market. pct_change:
                    pct_change_10d[i] = -1
            spy_df['pct_change_10d'] = pd.Series(pct_change_10d, index=spy_df.index)

            log.debug(f"results: {spy_df['pct_change_10d']}")

            for i in range(len(spy_df.index)):
                cur_datetime = spy_df.index[i]
                cur_market_status = spy_df['pct_change_10d']
                log.debug(f"{cur_datetime} market status over the next 10 days: {cur_market_status}")

                if cur_market_status == 1:
                    # TODO: handle
                    return

            return
        except Exception as err:
            log.warning(f"error backtesting stock symbol selection process: {err}", exc_info=True)


