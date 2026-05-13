# backtest.py
import asyncio
import MySQLdb
import logging
import threading
from core.config import config
from django.core.management.base import BaseCommand
from core.stratTestUtils import test_strategy

log = logging.getLogger(__name__)

'''
to run test:
$python3 src/manage.py backtest
'''
class Command(BaseCommand):
    help = ''

    def handle(self, *args, **options):
        try:
            test_strategy()

        except Exception as err:
            log.warning(f"error backtesting strategy: {err}", exc_info=True)


