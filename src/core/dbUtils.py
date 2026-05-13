import requests
import logging
import time
import MySQLdb
from django.db import connections
import pandas as pd

from core.config import config
from datetime import datetime, timedelta
from core.models import *
from core.utils import my_pct_change, cut_decimals, check_outliers, winsor_data, zscore, plot_standardized_data
log = logging.getLogger(__name__)


def clear_db():
    try:
        ensure_connection()
        LongStocks.objects.all().delete()
        ShortStocks.objects.all().delete()
        IntraDayData.objects.all().delete()
        QEarnings.objects.all().delete()
        MonthlyMacroData.objects.all().delete()
        QuarterlyMacroData.objects.all().delete()
        AnnualMacroData.objects.all().delete()

        return
    except Exception as err:
        log.error(f"error clearing db: {err}", exc_info=True)
        return
    
# makes sure we are connected to db - reinitializes connection if terminated
def ensure_connection():
    try:
        conn = connections['default']
        conn.close()
        conn.connect()
    except Exception as err:
        log.error(f"error getting mysql connection: {err}", exc_info=True)
        