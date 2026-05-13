# startup.py
import asyncio
from datetime import datetime, timedelta
import alpaca_trade_api
import pandas as pd
import MySQLdb
import logging
import threading

import pytz
from core.config import config
from django.core.management.base import BaseCommand
from core.intraDayUtils import get_latest_price

log = logging.getLogger(__name__)

'''
to run test:
$python3 src/manage.py buy_stock
'''
class Command(BaseCommand):
    help = ''

    def handle(self, *args, **options):
        try:
            url = config.alpaca.tradeUrl
            api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

            # first determine if market is open
            is_open_res = api.get_clock()
            log.debug(f"is open: {is_open_res}")
            is_open = is_open_res.is_open
            log.warning(f"is open equals: {is_open}")
            return
        except Exception as err:
            log.warn(f"error starting app: {err}", exc_info=True)


