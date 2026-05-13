'''
anything involving retrieving, analyzing, and cleaning company earnings reports
'''
import MySQLdb
import numpy as np
import requests
import logging
import time
import pytz
import threading
import finnhub

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from django.http import JsonResponse
from django.utils import timezone
from core.config import config
from core.dbUtils import ensure_connection
from datetime import datetime, timedelta
from core.models import QEarnings, QEarningsSentiment
from core.utils import my_pct_change, replace_inf, zscore, cut_decimals, winsor_data, log_iqr, check_outliers, exp_decay, plot_standardized_data
from core.sentimentUtils import get_earnings_sentiment, get_earnings_toneshift
from core.macroUtils import get_macro_indicators
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def get_earnings(symbol):
    get_earnings_data(symbol=symbol)
    # get_earnings_transcripts(symbol=symbol, retry_attempts=5)
    return

'''
for a stock symbol, get the list of earnings transcripts, get each transcript and run sentiment analysis on each
'''
def get_earnings_transcripts(symbol, retry_attempts):
    try:
        client = finnhub.Client(api_key=config.finnhub.apikey)

        # get list of transcripts 
        try:
            response = client.transcripts_list(symbol=symbol)
            time.sleep(float(config.time_buffer))
        except Exception as err:
            time.sleep(10)
            if retry_attempts > 0:
                log.error(f"error getting transcript list for {symbol} - retrying: {retry_attempts}")
                return get_earnings_transcripts(symbol=symbol, retry_attempts=(retry_attempts - 1))
            else:
                log.error(f"error getting transcript list for {symbol}: {err}", exc_info=True)
                return

        transcripts = response.get("transcripts")
        if transcripts is None:
            log.warning(f"{symbol} transcripts empty: {transcripts}")
            return
        
        # for each transcript, get the speech and run sentiment analysis
        transcript_dict = {
            "id": [],
            "time": [],
            "title": [],
            "transcript": [],
            "sentiment_score": [],
            "toneshift_score": [],
        }

        retry_attempts = 5
        i = 0
        while i < len(transcripts):
            transcript = transcripts[i]
            id = transcript.get("id")
            time_ = transcript.get("time")
            title = transcript.get("title")

            try:
                response = client.transcripts(id)
                time.sleep(float(config.time_buffer))
            except Exception as err:
                time.sleep(10) # give api time to recover
                if retry_attempts > 0:
                    retry_attempts -= 1
                    log.error(f"error getting transcript for {symbol} - {id} - retrying: {retry_attempts}")
                    continue
                else:
                    log.error(f"error getting transcript for {symbol} - {id} - skipping: {err}", exc_info=True)
                    i += 1
                    continue
        
            # reset the attempts if successful
            retry_attempts = 5
            speech_data = response.get("transcript")
            if speech_data is None:
                log.warning(f"empty speech data for {symbol} - {id}: skipping")
                i += 1
                continue


            # get all of the speech data and run sentiment analysis
            speech = ""
            for x in speech_data:
                log.info(f"speech list length: {len(x.get("speech"))}")
                log.info(f"speech: {x.get("speech")[0]}")
                speech += f"{x.get("name")}: {x.get("speech")[0]} \n"

            log.debug(f"final speech data: {speech}")
            sentiment_score = get_earnings_sentiment(symbol=symbol, txt=speech)
            time.sleep(float(config.time_buffer))
            toneshift_score = get_earnings_toneshift(symbol=symbol, txt=speech)

            if sentiment_score != None and toneshift_score != None:
                transcript_dict["id"].append(id)
                transcript_dict["time"].append(time_)
                transcript_dict["title"].append(title)
                transcript_dict["transcript"].append(speech)
                transcript_dict["sentiment_score"].append(sentiment_score)
                transcript_dict["toneshift_score"].append(toneshift_score)
            
            i += 1
        
        df = pd.DataFrame(transcript_dict, index=transcript_dict["time"]).sort_index()

        # get data from db to determine whats new
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_qearningssentiment WHERE symbol = '{symbol}';"
        df_db = pd.read_sql(query, conn)

        db_ids = set(df_db["id"])

        # save data in db
        for i in range(len(df["title"])):
            if df["id"][i] in db_ids:
                log.info(f"{symbol} earnings transcript already in db: {df["title"]}: skipping")
                continue
            
            QEarningsSentiment(
                symbol=symbol,
                id=df["id"][i],
                time=df["time"][i],
                title=df["title"][i],
                transcript=df["transcript"][i],
                sentiment_score=df["sentiment_score"][i],
                toneshift_score=df["toneshift_score"][i],
            ).save()

        conn.close()
        return
    except Exception as err:
        log.error(f"error getting earnings transcripts for {symbol}: {err}", exc_info=True)
        return
'''
Process: retrieve data - cut small numbers - get percent changes - winsor data - extend data - standardize - store data
TODO: infinity is trying to be saved - convert to real number to prevent errors
'''
def get_earnings_data(symbol):
    try:
        df, err = get_financial_statements(symbol=symbol, retry_attempts=3)

        if err != None:
            log.debug(f"error getting qearnings df for {symbol} - skipping")
            return

        if len(df.index) == 0:
            log.debug(f"qearnings df for {symbol} empty - skipping")
            return

        # check if any columns not present in q reports
        keys = ['totalAssets', 'totalLiabilities', 'totalEquity', 'currentAssets', 'currentLiabilities', 'cashEquivalents', 
                'netOperatingCashFlow', 'netInvestingCashFlow',  'netCashFinancingActivities', 'depreciationAmortization', 
                'netIncome', 'capex', 'totalOperatingExpense', "operatingMargin", 'netMargin', 'grossIncome', 'grossMargin', 'fcf', 'debt_equity_ratio', 
                'ebit', 'ebitda', 'revenue', 'eps', 'roa', 'roe', 'sharesOutstanding', 'bvps', 'netDebt', 'totalDebt', 'longTermDebt', 
                'totalDebtToEquity', 'currentRatio', 'quickRatio', 'cashRatio', 'totalRatio', 'pb', 'ev', 'commonStock', 'ebitPerShare']
        
        # if key not present, create and fill with None
        for key in keys:
            if key not in df.keys():
                log.debug(f"{key} not present - filling with None")
                df[key] = None
        
        # replace infinities with 99th percentile value to prevent sql error
        percentiles = df.replace([np.inf, -np.inf], np.nan).quantile(0.99)
        for col in df.columns:
            df[col] = df[col].replace(np.inf, percentiles[col])

        log.debug(f"earnings: current time: {datetime.now(tz=pytz.timezone('US/Eastern'))} | last earning time: {df.index[-1].astimezone(pytz.timezone('US/Eastern'))}")

        for i in range(len(df['revenue'])):
            ensure_connection() # ensure mysql connection
            QEarnings(        
                symbol=symbol,
                date=df.index[i],

                total_assets=df["totalAssets"][i] if not pd.isna(df['totalAssets'][i]) else None,
                total_liabilities=df["totalLiabilities"][i] if not pd.isna(df['totalLiabilities'][i]) else None,
                shareholders_equity=df["totalEquity"][i] if not pd.isna(df['totalEquity'][i]) else None,
                current_assets=df["currentAssets"][i] if not pd.isna(df['currentAssets'][i]) else None,
                current_liabilities=df["currentLiabilities"][i] if not pd.isna(df['currentLiabilities'][i]) else None,
                cash=df["cashEquivalents"][i] if not pd.isna(df['cashEquivalents'][i]) else None,

                operating_cash_flow=df["netOperatingCashFlow"][i] if not pd.isna(df['netOperatingCashFlow'][i]) else None,
                investing_cash_flow=df["netInvestingCashFlow"][i] if not pd.isna(df['netInvestingCashFlow'][i]) else None,
                financing_cash_flow=df["netCashFinancingActivities"][i] if not pd.isna(df['netCashFinancingActivities'][i]) else None,
                depreciation_amortization=df["depreciationAmortization"][i] if not pd.isna(df['depreciationAmortization'][i]) else None,
                net_income=df["netIncome"][i] if not pd.isna(df['netIncome'][i]) else None,
                cap_ex=df["capex"][i] if not pd.isna(df['capex'][i]) else None,

                operating_expenses=df["totalOperatingExpense"][i] if not pd.isna(df['totalOperatingExpense'][i]) else None,
                operating_margin=df["operatingMargin"][i] if not pd.isna(df['operatingMargin'][i]) else None,
                net_margin=df["netMargin"][i] if not pd.isna(df['netMargin'][i]) else None,
                gross_profit=df["grossIncome"][i] if not pd.isna(df['grossIncome'][i]) else None,
                gross_margin=df["grossMargin"][i] if not pd.isna(df['grossMargin'][i]) else None,
                free_cash_flow=df["fcf"][i] if not pd.isna(df['fcf'][i]) else None, 
                debt_equity_ratio=df["debt_equity_ratio"][i] if not pd.isna(df['debt_equity_ratio'][i]) else None,

                ebit=df["ebit"][i] if not pd.isna(df['ebit'][i]) else None,
                ebitda=df["ebitda"][i] if not pd.isna(df['ebitda'][i]) else None, 
                revenue=df["revenue"][i] if not pd.isna(df['revenue'][i]) else None,
                eps=df["eps"][i] if not pd.isna(df['eps'][i]) else None,
                roa=df["roa"][i] if not pd.isna(df['roa'][i]) else None, 
                roe=df["roe"][i] if not pd.isna(df['roe'][i]) else None, 
                shares_outstanding=df["sharesOutstanding"][i] if not pd.isna(df['sharesOutstanding'][i]) else None,
                bvps=df["bvps"][i] if not pd.isna(df['bvps'][i]) else None, 

                net_debt=df["netDebt"][i] if not pd.isna(df["netDebt"][i]) else None,
                total_debt=df["totalDebt"][i] if not pd.isna(df["totalDebt"][i]) else None,
                long_term_debt=df["longTermDebt"][i] if not pd.isna(df["longTermDebt"][i]) else None,
                total_debt_to_equity=df["totalDebtToEquity"][i] if not pd.isna(df["totalDebtToEquity"][i]) else None,

                cur_ratio=df["currentRatio"][i] if not pd.isna(df["currentRatio"][i]) else None,
                quick_ratio=df["quickRatio"][i] if not pd.isna(df["quickRatio"][i]) else None,
                cash_ratio=df["cashRatio"][i] if not pd.isna(df["cashRatio"][i]) else None,
                total_ratio=df['totalRatio'][i],

                pb=df["pb"][i] if not pd.isna(df["pb"][i]) else None,
                ev=df["ev"][i] if not pd.isna(df["ev"][i]) else None,

                commonStock=df['commonStock'][i],
                fcf=df['fcf'][i],
                ebitPerShare=df['ebitPerShare'][i],

            ).save()
    except Exception as err:
        log.error(f"error getting quarterly data: {err}", exc_info=True)
        return
    
def get_earnings_from_db(symbol, tradingFrame, withTranscripts):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_qearnings WHERE symbol = '{symbol}';"
        df = pd.read_sql(query, conn)

        # if using earnings transcripts
        if withTranscripts:
            query = f"SELECT * FROM core_qearningssentiment WHERE symbol = '{symbol}';"
            df_transcripts = pd.read_sql(query, conn)

            if len(df_transcripts.index):
                log.info("transcripts df is empty - data either not retrieved or not available")
                return None, None
            
            df_transcripts = df_transcripts.drop(columns=['uid', 'symbol', 'title', 'id', 'transcript'])

            # shift timestamps to EST/EDT
            df_transcripts = df_transcripts.set_index('time').sort_index()
            df_transcripts.index = df_transcripts.index.tz_localize('UTC')
            df_transcripts.index = df_transcripts.index.tz_convert('US/Eastern')

        # check if data is null
        if len(df.index) == 0:
            log.info("earnings df is empty - data either not retrieved or not available")
            return None, None

        df = df.drop(columns=['uid', 'symbol'])
        
        # shift timestamps to EST/EDT
        df = df.set_index('date').sort_index()
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')
        
        df = get_macro_indicators(df=df)
        # mark every time an earnings report was released - should be every index before extension
        df['report_released'] = pd.Series([1] * len(df.index), index=df.index)

        # save copy of raw data 
        df_raw = df.copy()
        
        # extend data to fit hourly scale - static data
        start_time = df.index.min()

        # if start time is after a certain date, return - not enough data
        eastern = pytz.timezone('US/Eastern')
        cutoff_date = datetime.combine(config.cutoff_date, datetime.min.time())
        cutoff_date = eastern.localize(cutoff_date)
        if start_time > cutoff_date:
            log.info(f"not enough earnings data for {symbol}: start date: {start_time} | cutoff: {cutoff_date}")
            return None, None

        # have data end at the most recent hour
        current_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))
        # current_time = timezone.now().astimezone(tz=pytz.timezone('UTC'))

        # extend based on if trading frame is daily or intraday
        if tradingFrame == 'intraday':
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_hour)

            new_time_index = pd.date_range(start=start_time, end=end_time, freq='H')

            # if using earnings transcripts
            if withTranscripts:
                # transcript data
                df_transcripts.index = df_transcripts.index.ceil('H')
                start_time_transcripts = df_transcripts.index.min()
                new_time_index_transcripts = pd.date_range(start=start_time_transcripts, end=end_time, freq='H')
        elif tradingFrame == 'daily':
            current_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_day)

            new_time_index = pd.date_range(start=start_time, end=end_time, freq='D')

            # if using earnings transcripts
            if withTranscripts:
                # transcript data
                df_transcripts.index = df_transcripts.index.floor('D')
                start_time_transcripts = df_transcripts.index.min()
                new_time_index_transcripts = pd.date_range(start=start_time_transcripts, end=end_time, freq='D')
        else:
            log.error(f"invalid trading frame - cannot get earnings data")
            return

        df_extended = df.reindex(new_time_index)
        log.debug(f"{symbol} earnings: tradingFrame: {tradingFrame} | index: {df_extended.index}")

        # if using earnings transcripts
        if withTranscripts:
            df_transcripts_extended = df_transcripts.reindex(new_time_index_transcripts)
            log.debug(f"{symbol} earnings: tradingFrame: {tradingFrame} | transcript index: {df_transcripts_extended.index}")

            # if using transcripts, join data with base dataframe
            df_extended = df_extended.join(df_transcripts_extended)
            log.debug(f"joined earnings db: {df_extended}")

        # get exp decay
        df_extended['report_released'] = df_extended['report_released'].fillna(0)
        log.debug(f"extended report released: {df_extended['report_released']}")
        df = df_extended.ffill().fillna(0)

        df = df.add_suffix('_qearnings')
        df_raw = df_raw.add_suffix('_qearnings_raw')

        conn.close()
        return df_raw, df
    except Exception as err:
        log.error(f"error getting quarterly reports: {err}", exc_info=True)
        return None, None


def get_financials(freq):
    try:
        symbol = config.symbol
        start_date = config.start_date
        end_date = config.end_date
        client = finnhub.Client(api_key=config.finnhub.apikey)

        financials = {
            "startDate": [],
            "endDate": [],
            "date": [],
            "quarter": [],
            "total_assets": [],
            "total_liabilities": [],
            "shareholders_equity": [],
            "current_assets": [],
            "current_liabilities": [],
            "cash": [], 
            # "cash_equity": [], #
            "inventory": [],  

            "operating_cash_flow": [], 
            "investing_cash_flow": [], 
            "financing_cash_flow": [], 
            "depreciation_amortization": [],
            "net_income": [],
            "cap_ex": [], 

            "operating_income": [], 
            "non_operating_income": [],
            "operating_expenses": [], 
            "operating_margin": [], 
            "net_margin": [], 
            "gross_profit": [], 
            "gross_margin": [], 
            "free_cash_flow": [],
            "debt_equity_ratio": [],

            "eps": [],
            "revenue": [],
            "shares_outstanding": [],
            "ebit": [], 
            "ebitda": [], 
            "bvps": [],
            "roa": [],
            "roe": [],
        }

        while start_date < end_date:
            # to_ = start_date + timedelta(days=20)
            # if (to_ > end_date):
            #     to_ = end_date

            response = client.financials_reported(symbol=symbol, freq=freq, _from=start_date.strftime("%Y-%m-%d"), to=end_date.strftime("%Y-%m-%d"))
            results = response.get("data")
            for result in results:
                operating_income = non_operating_income = operating_expenses = gross_profit = eps = revenue = shares_outstanding = None
                total_assets = total_liabilities = shareholder_equity = cur_assets = cur_liabilities = cash = inventory_net = None
                operating_cash_flow = invest_cash_flow = fin_cash_flow = deprec_amor = net_income = cap_ex = None

                financials["startDate"].append(result.get("startDate"))
                financials["endDate"].append(result.get("endDate"))
                financials["date"].append(result.get("filedDate"))
                financials["quarter"].append(result.get("quarter"))

                # total_assets, total_liabilities, shareholder_equity, cur_assets, cur_liabilities, cash, inventory_net = None
                # operating_cash_flow, invest_cash_flow, fin_cash_flow, operating_income, non_operating_income, operating_expenses, operating_margin, net_margin, gross_profit = None
                # gross_margin, ebit, ebitda, revenue, eps, roa, roe = None
                # deprec_amor, interest_exp, net_income, free_cash_flow, cap_ex, debt_equity_ratio, bvps, shares_outstanding = None
                # cash equity, 
                report = result.get("report")
                bs = report.get("bs")
                cf = report.get("cf")
                ic = report.get("ic")
                log.debug(f"balance sheet len: {len(bs)}")
                # getting fields within balance sheet
                for field in bs:
                    concept = field.get("concept")
                    value = field.get("value")
                    if concept == "us-gaap_Assets": # total assets
                        total_assets = value
                    elif concept == "us-gaap_Liabilities": # total liabilities
                        total_liabilities = value
                    elif concept == "us-gaap_StockholdersEquity": # total shareholders equity
                        shareholder_equity = value
                    elif concept == "us-gaap_AssetsCurrent": # total current assets
                        cur_assets = value
                    elif concept == "us-gaap_LiabilitiesCurrent": # total current liabilities
                        cur_liabilities = value
                    elif concept == "us-gaap_CashAndCashEquivalentsAtCarryingValue": # cash
                        cash = value
                    elif concept == "us-gaap_InventoryNet": # net inventory
                        inventory_net = value
                for field in cf:
                    concept = field.get("concept")
                    value = field.get("value")
                    if concept == "us-gaap_NetCashProvidedByUsedInOperatingActivities":
                        operating_cash_flow = value
                    elif concept == "us-gaap_NetCashProvidedByUsedInInvestingActivities":
                        invest_cash_flow = value
                    elif concept == "us-gaap_NetCashProvidedByUsedInFinancingActivities":
                        fin_cash_flow = value           
                    elif concept == "us-gaap_DepreciationDepletionAndAmortization":
                        deprec_amor = value         
                    elif concept == "us-gaap_NetIncomeLoss":
                        net_income = value
                    elif concept == "us-gaap_PaymentsToAcquirePropertyPlantAndEquipment":
                        cap_ex = value
                for field in ic:
                    concept = field.get("concept")
                    value = field.get("value")
                    if concept == "us-gaap_OperatingIncomeLoss":
                        operating_income = value
                    elif concept == "us-gaap_NonoperatingIncomeExpense":
                        non_operating_income = value
                    elif concept == "us-gaap_OperatingExpenses":
                        operating_expenses = value
                    elif concept == "us-gaap_GrossProfit":
                        gross_profit = value
                    elif concept == "us-gaap_EarningsPerShareBasic":
                        eps = value
                    elif concept == "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax":
                        revenue = value
                    elif concept == "us-gaap_WeightedAverageNumberOfSharesOutstandingBasic":
                        shares_outstanding = value

                # calculate additional indicators
                operating_margin = net_margin = gross_margin = None
                free_cash_flow = debt_equity_ratio = bvps = None
                ebit = ebitda = roa = roe = None

                if revenue != None:
                    operating_margin = operating_income / revenue 
                    net_margin = net_income / revenue 
                    gross_margin = gross_profit / revenue 

                if operating_cash_flow != None and cap_ex != None:
                    free_cash_flow = operating_cash_flow - cap_ex   

                debt_equity_ratio = total_liabilities / shareholder_equity
                bvps = shareholder_equity / shares_outstanding
                ebit = operating_income + non_operating_income
                if deprec_amor != None:
                   ebitda = ebit + deprec_amor 

                roa = net_income / total_assets
                roe = net_income / shareholder_equity

                financials["total_assets"].append(total_assets)
                financials["total_liabilities"].append(total_liabilities)
                financials["shareholders_equity"].append(shareholder_equity)
                financials["current_assets"].append(cur_assets)
                financials["current_liabilities"].append(cur_liabilities)
                financials["cash"].append(cash)
                financials["inventory"].append(inventory_net)
                financials["operating_cash_flow"].append(operating_cash_flow)
                financials["investing_cash_flow"].append(invest_cash_flow)
                financials["financing_cash_flow"].append(fin_cash_flow)
                financials["depreciation_amortization"].append(deprec_amor)
                financials["net_income"].append(net_income)
                financials["cap_ex"].append(cap_ex)
                financials["operating_income"].append(operating_income)
                financials["non_operating_income"].append(non_operating_income)
                financials["operating_expenses"].append(operating_expenses)
                financials["operating_margin"].append(operating_margin)
                financials["net_margin"].append(net_margin)
                financials["gross_profit"].append(gross_profit)
                financials["gross_margin"].append(gross_margin)
                financials["free_cash_flow"].append(free_cash_flow)
                financials["debt_equity_ratio"].append(debt_equity_ratio)
                financials["eps"].append(eps)
                financials["revenue"].append(revenue)
                financials["shares_outstanding"].append(shares_outstanding)
                financials["ebit"].append(ebit)
                financials["ebitda"].append(ebitda)
                financials["bvps"].append(bvps)
                financials["roa"].append(roa)
                financials["roe"].append(roe)
            
            # start_date = to_ 
            start_date = end_date
        for key, value in financials.items():
            log.debug(f"key: {key} - value length: {len(value)} - value 0: {value[0]}")

        return pd.DataFrame(financials, index=financials["date"]).sort_index()
    except Exception as err:
        log.error(f"error getting company financial data: {err}", exc_info=True)
        return
    

# makes api calls to get data from quarterly reports 
# @param retry_attempts - if api call fails, retry function
# @return df, error - error = None if successful
def get_financial_statements(symbol, retry_attempts):
    try:
        if retry_attempts == 0:
            log.error(f"error getting financial statements for {symbol} - retry attempts = 0")
            return None, "error"
        
        client = finnhub.Client(api_key=config.finnhub.apikey)

        # financials
        log.debug("getting financials")
        # make api calls and get df's for each quarterly report statement
        df_bs, bs_err = get_fin_by_statement(symbol=symbol, statement='bs', freq='quarterly', retry_attempts=3)
        df_ic, ic_err = get_fin_by_statement(symbol=symbol, statement='ic', freq='quarterly', retry_attempts=3)
        df_cf, cf_err = get_fin_by_statement(symbol=symbol, statement='cf', freq='quarterly', retry_attempts=3)

        if bs_err != None or ic_err != None or cf_err != None:
            log.error("warning: unable to get fin by statement - skipping")
            return None, "error"

        # drop duplicate columns
        dup_cols = df_bs.columns.intersection(df_ic.columns)
        if (len(dup_cols) > 0):
            df_ic = df_ic.drop(columns=dup_cols)
        dup_cols = df_bs.columns.intersection(df_cf.columns)
        if (len(dup_cols) > 0):
            df_cf = df_cf.drop(columns=dup_cols)
        dup_cols = df_cf.columns.intersection(df_ic.columns)
        if (len(dup_cols) > 0):
            df_ic = df_ic.drop(columns=dup_cols)
    
        # merge all df's into one final df    
        df_fin = df_bs.join(df_ic, how='inner').join(df_cf, how='inner')

        # basic financials
        log.debug("getting basic financials")

        try:
            basic_fin_resp = client.company_basic_financials(symbol, 'all')
        except Exception as err: # 500 error - retry
            time.sleep(10) # give server time
            retry_attempts -= 1
            return get_financial_statements(symbol=symbol, retry_attempts=retry_attempts)
        
        time.sleep(float(config.time_buffer))
        basic_fin_results = basic_fin_resp.get("series").get("quarterly")

        # create small df's for each key, and join with final df
        basic_keys = ["cashRatio", "currentRatio", "ebitPerShare", "eps", "ev", "grossMargin", "netDebtToTotalCapital", "netDebtToTotalEquity", "netMargin", "operatingMargin", "pb", "pretaxMargin", "quickRatio", "salesPerShare", "sgaToSale", "totalDebtToEquity", "totalDebtToTotalAsset", "totalDebtToTotalCapital", "totalRatio"]

        for key in basic_keys:
            log.debug(f"cur key: {key}")
            # get list of dictionaries containing date and val for the individual key
            indiv_data = basic_fin_results.get(key)
            basic_data_dict = {
                'period': [],
                key: [],
            }

            if indiv_data is None:
                continue

            # iterate through list of dicts and save period and vals
            for i in indiv_data:
                date = datetime.strptime(i.get('period'), "%Y-%m-%d")
                date = pytz.timezone('US/Eastern').localize(date) # TODO: verify this
                date = date.astimezone(pytz.timezone('UTC'))
                basic_data_dict['period'].append(date)
                basic_data_dict[key].append(i.get('v'))

            # create df and merge with final df
            df_temp = pd.DataFrame(basic_data_dict)
            df_temp = df_temp.set_index(df_temp['period']).sort_index()
            df_temp = df_temp.drop(columns='period')

            # merge with final df - fill nil values with 0
            # TODO: forward fill nil or fill with 0?
            # TODO: left or outer join?
            df_fin = df_fin.join(df_temp, how='outer').ffill().fillna(0)

        if 'netIncome' in df_fin and 'totalAssets' in df_fin:
            df_fin['roa'] = df_fin['netIncome'] / df_fin['totalAssets']
        if 'netIncome' in df_fin and 'totalEquity' in df_fin:
            df_fin['roe'] = df_fin['netIncome'] / df_fin['totalEquity']
        if 'totalEquity' in df_fin and 'sharesOutstanding' in df_fin:
            df_fin['bvps'] = df_fin['totalEquity'] / df_fin['sharesOutstanding']
        if 'ebit' in df_fin and 'depreciationAmortization' in df_fin:
            df_fin['ebitda'] = df_fin['ebit'] + df_fin['depreciationAmortization']
        if 'totalLiabilities' in df_fin and 'totalEquity' in df_fin:
            df_fin['debt_equity_ratio'] = df_fin['totalLiabilities'] / df_fin['totalEquity']


        log.debug(df_fin.keys())

        return df_fin.sort_index(), None
    except Exception as err:
        log.error(f"error getting financial statements: {err}", exc_info=True)
        return None, "error"
    
# takes stock ticker, statement (bs, ic, cf), and frequency (quarterly, annual) and returns df if successful
# @return df, error - error = None if successful
def get_fin_by_statement(symbol, statement, freq, retry_attempts):
    try:
        if retry_attempts == 0:
            log.error(f"warning: unable to get fin by statment for {symbol} - retry attempts = 0")
            return None, "error"

        client = finnhub.Client(api_key=config.finnhub.apikey)

        financials = {}
        try:
            response = client.financials(symbol=symbol, statement=statement, freq=freq)
        except Exception:
            log.info(f"warning: unable to get fin by statement for {symbol} - retrying")
            time.sleep(10) # give server time
            return get_fin_by_statement(symbol=symbol, statement=statement, freq=freq, retry_attempts=retry_attempts-1)
        time.sleep(float(config.time_buffer))
        results = response.get("financials")
        for key in results[0].keys():
            financials[key] = [x.get(key) for x in results]
        
        df = pd.DataFrame(financials, index=financials['period']).sort_index()
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize('US/Eastern')
        df.index = df.index.tz_convert('UTC')

        df = df.drop(columns='period')
        return df, None
    except Exception as err:
        log.error(f"warning: unable to get fin by statement type for {symbol}: {err}")
        return None, "error"

def realTimeQuarterlyReportData(symbols):
    try:
        for symbol in symbols:
            log.debug(f"getting current quarterly reports for {symbol} at {datetime.now().time()}")
            # clear entries and db and retrieve new ones - lazy but doesnt take much time
            QEarnings.objects.filter(symbol=symbol).delete()

            # get new data
            get_earnings(symbol=symbol, start_date=config.start_date)
    except Exception as err:
        log.error(f"error getting q reports: {err}", exc_info=True)
