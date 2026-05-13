# continuousNews.py
import asyncio
import MySQLdb
import logging
import threading

import finnhub
from core.config import config
from django.core.management.base import BaseCommand
from core.control import startup
from core.newsUtils import get_news_data
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

'''
to run test:
$python3 src/manage.py continuousNews
'''
class Command(BaseCommand):
    help = ''

    def handle(self, *args, **options):
        try:
            # make api call to get most symbols in us
            client = finnhub.Client(api_key=config.finnhub.apikey)
            mics = ['XNYS','XASE','BATS','XNAS']
            symbols = []
            for mic in mics:
                # get list of symbols
                try:
                    response = client.stock_symbols(exchange='US', mic=mic, currency='USD')
                except Exception:
                        log.error(f"error: unable to get symbols")
                        return
                    
                for stock in response:
                    symbols.append(stock.get('symbol'))

            # remove any stocks with periods - these are substocks
            symbols = list(filter(lambda x: "." not in x, symbols))

            # filter out stocks with below 10 billion
            log.debug(f"number of symbols: {len(symbols)}")

            start_date = config.start_date
            end_date = datetime.today().date()
            log.debug(f"getting news data from {start_date} to {end_date}")
            for symbol in symbols:
                log.debug(f"getting news data for {symbol}")
                get_news_data(symbol=symbol, start_date=start_date, end_date=end_date)
        except Exception as err:
            log.warn(f"error getting news data: {err}", exc_info=True)


