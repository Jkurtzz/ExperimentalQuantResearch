# update_stock_data.py
import pandas as pd
import MySQLdb
import logging
import threading
from core.config import config
from django.core.management.base import BaseCommand
from core.utils import plot_standardized_data
from core.dailyUtils import get_daily_data
from core.earningsUtils import get_earnings
from core.insiderUtils import get_insider_transactions
from core.intraDayUtils import get_intra_day_data
from core.macroUtils import get_macro
from core.newsUtils import get_news_data
from core.pressUtils import get_press_releases
from core.socialUtils import get_social_media_data

log = logging.getLogger(__name__)

'''
to run test:
$python3 src/manage.py update_stock_data
'''
class Command(BaseCommand):
    help = 'Updates the stock data'

    def handle(self, *args, **options):
        try:
            fetch_stock_data()  # Call the fetch_stock_data function
            log.debug("retrieved stock data; exporting to csv")
            conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)

            # stock symbols
            query = "SELECT * FROM core_stocks;"
            df_symbols = pd.read_sql(query, conn)
            df_symbols.to_csv('symbols.csv', index=False)

            query = "SELECT * FROM core_intradaydata;"
            df_intraday = pd.read_sql(query, conn)
            df_intraday.to_csv('intradaydata.csv', index=False)

            query = "SELECT * FROM core_dailydata;"
            df = pd.read_sql(query, conn)
            df.to_csv('dailydata.csv', index=False)
            
            # news articles
            query = "SELECT * FROM core_finnarticles;"
            df_news = pd.read_sql(query, conn)
            df_news.to_csv('finnarticles.csv', index=False)

            # press releases
            query = "SELECT * FROM core_pressreleases;"
            df = pd.read_sql(query, conn)
            df.to_csv('pressreleases.csv', index=False)

            # insider transactions
            query = "SELECT * FROM core_insidertransactions;"
            df = pd.read_sql(query, conn)
            df.to_csv('insidertransactions.csv', index=False)

            query = "SELECT * FROM core_socialmedia;"
            df_social = pd.read_sql(query, conn)
            df_social.to_csv('socialmedia.csv', index=False)

            # quarterly earnings
            query = "SELECT * FROM core_qearnings;"
            df = pd.read_sql(query, conn)
            df.to_csv('qearningsreports.csv', index=False)

            # monthly macro data
            query = "SELECT * FROM core_monthlymacrodata;"
            df = pd.read_sql(query, conn)
            df.to_csv('monthlymacrodata.csv', index=False)

            # quarterly macro data
            query = "SELECT * FROM core_quarterlymacrodata;"
            df = pd.read_sql(query, conn)
            df.to_csv('quarterlymacrodata.csv', index=False)

            # annual macro data
            query = "SELECT * FROM core_annualmacrodata;"
            df = pd.read_sql(query, conn)
            df.to_csv('annualmacrodata.csv', index=False)

        except Exception as err:
            log.warn(f"error retrieving and saving stock data: {err}", exc_info=True)


def fetch_stock_data():
    news_thread = threading.Thread(target=get_news_data)
    press_thread = threading.Thread(target=get_press_releases)
    intra_thread = threading.Thread(target=get_intra_day_data)
    daily_thread = threading.Thread(target=get_daily_data)
    insider_thread = threading.Thread(target=get_insider_transactions)
    social_thread = threading.Thread(target=get_social_media_data)
    earnings_thread = threading.Thread(target=get_earnings)
    macro_thread = threading.Thread(target=get_macro)

    # news_thread.start()
    # press_thread.start()
    # intra_thread.start()
    # daily_thread.start()
    # insider_thread.start()
    # social_thread.start()
    # earnings_thread.start()
    # macro_thread.start()

    # news_thread.join()
    # press_thread.join()
    # intra_thread.join()
    # daily_thread.join()
    # insider_thread.join()
    # social_thread.join()
    # earnings_thread.join()
    # macro_thread.join()
    return