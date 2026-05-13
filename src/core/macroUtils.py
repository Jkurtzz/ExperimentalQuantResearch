'''
anything involving retrieving, analyzing, and cleaning macroeconomic data
'''
import MySQLdb
import requests
import logging
import pytz
import threading
import time
import finnhub
import pandas as pd
import pandas_ta as ta
import numpy as np


import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
from django.http import JsonResponse
from django.utils import timezone
from core.config import config
from datetime import datetime, timedelta
from core.models import MonthlyMacroData, QuarterlyMacroData, AnnualMacroData
from core.utils import my_pct_change, replace_inf, zscore, cut_decimals, winsor_data, log_iqr, check_outliers, exp_decay
from core.dbUtils import ensure_connection
from sklearn.preprocessing import StandardScaler


log = logging.getLogger(__name__)

'''
process: get data - get percent changes - winsor data - interpolate data - zscore data - store
'''
def get_macro(start_date):
    try:
        start_date = pd.Timestamp(start_date).tz_localize('US/Eastern')
        get_monthly_data(start_date=start_date)
        get_quarterly_data(start_date=start_date)
        get_annual_data(start_date=start_date)
        return
    except Exception as err:
        log.error(f"error getting macro: {err}", exc_info=True)

def get_monthly_data(start_date):
    try: 
        client = finnhub.Client(api_key=config.finnhub.apikey)
        # create base df to add to later - starting with cpi
        cpi_dict = {
            'date': [],
            'cpi': [],
        }
        try:
            response = client.economic_data('MA-USA-678073')    
        except Exception:
            log.warning("error trying to get econ data - retrying")
            time.sleep(10)
            response = client.economic_data('MA-USA-678073') 

        data = response['data']
        for entry in data:
            date = datetime.strptime(entry.get('date'), "%Y-%m-%d")
            date = pytz.timezone('US/Eastern').localize(date) # TODO: verify this
            date = date.astimezone(pytz.timezone('UTC'))
            if date > start_date:
                cpi_dict['date'].append(date)
                cpi_dict['cpi'].append(entry.get('value'))
        df_macro = pd.DataFrame(cpi_dict)
        df_macro = df_macro.set_index('date').sort_index()
        
        # get remaining data and concat with final df    

        # standard monthly data that doesnt need to be cleaned
        macro_data = {
            # 'cpi': 'MA-USA-678073',
            'core_consumer_prices': 'MA-USA-67678073',
            'inflation_rate': 'MA-USA-67807367',
            'inflation_rate_mom': 'MA-USA-6780736777',
            # 'ism_nyi': 'MA-USA-738377788973',
            'chicago_pmi': 'MA-USA-67807773',
            # 'composite_pmi': 'MA-USA-6777807773',
            'non_manufacturing_pmi': 'MA-USA-7877807773',
            'industrial_production': 'MA-USA-7380',
            'industrial_production_mom': 'MA-USA-7380777977',
            # 'manufacturing_pmi': 'MA-USA-77807773',
            'unemployment_rate': 'MA-USA-857882',
            'non_farm_payrolls': 'MA-USA-787080',
            'adp_employment_change': 'MA-USA-656880',
            'labor_force_participation_rate': 'MA-USA-76708082',
            'avg_hourly_earnings': 'MA-USA-69658278',
            # 'avg_hourly_wages': 'MA-USA-87657169',
            # 'fed_funds_rate': 'MA-USA-7382',
            # 'avg_monthly_prime_lending_rate': 'MA-USA-667682',
            'bal_of_trade': 'MA-USA-667984',
            'exports': 'MA-USA-6988867976',
            'imports': 'MA-USA-7377867976',
            'consumer_sentiment': 'MA-USA-67786778',
            'nfib_boi': 'MA-USA-78707366', # bussiness optimism index
            'ibd_eoi': 'MA-USA-798084', # economic optimism index
            'existing_home_sales': 'MA-USA-697283',
            'new_home_sales': 'MA-USA-787283',
            'housing_starts': 'MA-USA-72838484',
            'pending_home_sales': 'MA-USA-807283',
            'housing_price_index_mom_change': 'MA-USA-72798583',
            'ism_pmi': 'MA-USA-6667797870', 
        }

        # for each macro key and code, get data, save as temp df, concat with final df
        for key, code in macro_data.items():
            code_dict = {
                'date': [],
                key: [],
            }

            try:
                response = client.economic_data(code)
            except Exception:
                log.warning("error trying to get econ data - retrying")
                time.sleep(10)
                response = client.economic_data(code)

            time.sleep(float(config.time_buffer))
            data = response['data']
            for entry in data:
                date = datetime.strptime(entry.get('date'), "%Y-%m-%d")
                date = pytz.timezone('US/Eastern').localize(date) # TODO: verify this
                date = date.astimezone(pytz.timezone('UTC'))
                if date > start_date:
                    code_dict['date'].append(date)
                    code_dict[key].append(entry.get('value'))
            df_temp = pd.DataFrame(code_dict)
            df_temp = df_temp.set_index('date').sort_index()

            # concat with final df
            df_macro = pd.concat([df_macro, df_temp], axis=1)
            
        # TODO: add support for end-of-month and cummulative data
        # # same thing for end-of-month data
        # # not monthly data so grabbing last value of each month
        # # api call to each code, save only last value of each month
        # eom_data = {
        #     'three_month_interbank_rate': 'MA-USA-73667982',
        #     # 'goverment_bond_10y': 'MA-USA-71897668', # cant be used - not up to date
        #     # 'djia': 'MA-USA-777584', # dow jones industrial average # cant be used - not up to date
        # }

        # # set date field 
        # macro_data['date'] = date

        # # get non monthly end of month data and filter
        # if macro_data['date']:
        #     for key, code in eom_data.items():
        #         response = client.economic_data(code)
        #         time.sleep(float(config.time_buffer))
        #         # set dict to list to add to 
        #         eom_data[key] = []
        #         data = response['data']
        #         # filter out extra dates and keep last of each month
        #         cur_date = start_date
        #         while cur_date <= datetime.strptime(macro_data['date'][-1], "%Y-%m-%d"):
        #             cur_month = [x for x in data if datetime.strptime(x['date'], "%Y-%m-%d").year == cur_date.year and datetime.strptime(x['date'], "%Y-%m-%d").month == cur_date.month]
        #             # check if month is empty 
        #             if cur_month:
        #                 eom_data[key].append(cur_month[-1]['value'])
        #             cur_date += relativedelta(months=1)
        #     # move cleaned data to macro data 
        #     macro_data['three_month_interbank_rate'] = eom_data['three_month_interbank_rate']

        # # not monthly so summing data within month
        # cummulative_data = {
        #     'initial_jobless_claims': 'MA-USA-74677677',
        #     'continuing_jobless_claims': 'MA-USA-677467',
        #     'api_crude_oil_stock_change': 'MA-USA-6567798367',
        #     'crude_oil_stock_change': 'MA-USA-67798367',
        #     'natural_gas_stock_change': 'MA-USA-786584718367',
        #     'baker_hughes_crude_oil_rigs': 'MA-USA-677982',
        # }
        # if macro_data['date']:
        #     for key, code in cummulative_data.items():
        #         response = client.economic_data(code)
        #         time.sleep(float(config.time_buffer))
        #         # set dict to list to add to 
        #         cummulative_data[key] = []
        #         data = response['data']
        #         # filter out extra dates and keep last of each month
        #         cur_date = start_date
        #         # get sum of each month
        #         while cur_date <= datetime.strptime(macro_data['date'][-1], "%Y-%m-%d"):
        #             cur_month = [x['value'] for x in data if datetime.strptime(x['date'], "%Y-%m-%d").year == cur_date.year and datetime.strptime(x['date'], "%Y-%m-%d").month == cur_date.month]
        #             # check if month is empty 
        #             if cur_month:
        #                 # get sum of each month's data
        #                 cummulative_data[key].append(sum(cur_month))
        #             cur_date += relativedelta(months=1)
        #         # move data to macro data
        #         macro_data[key] = cummulative_data[key]

        # # some data slightly not up to date so add None in spots where its behind
        # # for exmaple, some rows latest data is july 31 while others is aug 31 so we enter None to signify august is not in yet
        # for key, data in macro_data.items():
        #     while len(data) < len(macro_data['date']):
        #         data.append(None) 

        # cpi values are raw - need to be pct change
        df_macro['cpi'] = df_macro['cpi'].pct_change()

        # replace NaN's with None so it can be stored in db
        df_macro = df_macro.replace({np.nan: None})

        # original data
        for i in range(len(df_macro['cpi'])):
            ensure_connection() # ensure mysql connection
            MonthlyMacroData(
                date=df_macro.index[i],

                cpi=df_macro['cpi'][i],
                core_consumer_prices=df_macro['core_consumer_prices'][i],
                inflation_rate=df_macro['inflation_rate'][i],
                inflation_rate_mom=df_macro['inflation_rate_mom'][i],
                chicago_pmi=df_macro['chicago_pmi'][i],
                non_manufacturing_pmi=df_macro['non_manufacturing_pmi'][i],
                industrial_production=df_macro['industrial_production'][i],
                industrial_production_mom=df_macro['industrial_production_mom'][i],
                unemployment_rate=df_macro['unemployment_rate'][i],
                non_farm_payrolls=df_macro['non_farm_payrolls'][i],
                adp_employment_change=df_macro['adp_employment_change'][i],
                labor_force_participation_rate=df_macro['labor_force_participation_rate'][i],
                avg_hourly_earnings=df_macro['avg_hourly_earnings'][i],
                bal_of_trade=df_macro['bal_of_trade'][i],
                exports=df_macro['exports'][i],
                imports=df_macro['imports'][i],
                consumer_sentiment=df_macro['consumer_sentiment'][i],
                nfib_boi=df_macro['nfib_boi'][i],
                ibd_eoi=df_macro['ibd_eoi'][i],
                existing_home_sales=df_macro['existing_home_sales'][i],
                new_home_sales=df_macro['new_home_sales'][i],
                housing_starts=df_macro['housing_starts'][i],
                pending_home_sales=df_macro['pending_home_sales'][i],
                housing_price_index_mom_change=df_macro['housing_price_index_mom_change'][i],
                ism_pmi=df_macro['ism_pmi'][i],
                # three_month_interbank_rate=df_macro['three_month_interbank_rate'][i],
                # initial_jobless_claims=df_macro['initial_jobless_claims'][i],
                # continuing_jobless_claims=df_macro['continuing_jobless_claims'][i],
                # api_crude_oil_stock_change=df_macro['api_crude_oil_stock_change'][i],
                # crude_oil_stock_change=df_macro['crude_oil_stock_change'][i],
                # natural_gas_stock_change=df_macro['natural_gas_stock_change'][i],
                # baker_hughes_crude_oil_rigs=df_macro['baker_hughes_crude_oil_rigs'][i],
            ).save()
    except Exception as err:
        log.error(f"error getting macro econ data: {err}", exc_info=True)
        return

# gets macro data from db based on freq (monthly, quarterly, annual)
def get_macro_from_db_by_freq(freq, tradingFrame):
    try:
        conn = MySQLdb.connect(host=config.database.host, user=config.database.user, passwd=config.database.password, db=config.database.name)
        query = f"SELECT * FROM core_{freq}macrodata;"
        df = pd.read_sql(query, conn)

        df = df.drop(columns=['uid'])

        # shift timestamps to EST/EDT
        df = df.set_index('date').sort_index()
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')

        # get indicators
        df = get_macro_indicators(df=df)

        df_raw = df.copy()
        
        # extend data to fit hourly scale - static data
        start_time = df.index.min()
        # have data end at the most recent hour
        current_time = timezone.now().astimezone(tz=pytz.timezone('US/Eastern'))

        if tradingFrame == 'intraday':
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_hour)
            new_time_index = pd.date_range(start=start_time, end=end_time, freq='H')
        elif tradingFrame == 'daily':
            current_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = pd.Timestamp(current_day)
            new_time_index = pd.date_range(start=start_time, end=end_time, freq='D')
        else:
            log.error(f"invalid trading frame - cannot get macro data")
            return

        df_extended = df.reindex(new_time_index)
        log.debug(f"macro: tradingFrame: {tradingFrame} | index: {df_extended.index}")

        # static data
        df_static = df_extended.ffill()

        # get percent changes
        df_pct = my_pct_change(df_static)

        # get exp decay
        df_exp = exp_decay(df=df_extended, rate=config.finnhub.macro.decay)

        conn.close()
        return df_raw, df_exp
    except Exception as err:
        log.error(f"error getting macro econ data from db: {err}", exc_info=True)
        return None, None

def get_macro_from_db(tradingFrame):
    try:
        # NOTE: consider separating the three types - losing data when we join
        df_monthly_macro, df_monthly_macro_exp = get_macro_from_db_by_freq(freq='monthly', tradingFrame=tradingFrame)
        df_quarterly_macro, df_quarterly_macro_exp = get_macro_from_db_by_freq(freq='quarterly', tradingFrame=tradingFrame)
        # NOTE: removing annual data for now - doesnt add enough value for how much gets removed
        # df_annual_macro, df_annual_macro_exp = get_macro_from_db_by_freq(freq='annual', tradingFrame=tradingFrame)

        # we want only 1 macro df for each
        df_macro_exp = df_monthly_macro_exp.join(df_quarterly_macro_exp, how='inner')
        df_macro_exp = df_macro_exp.add_suffix('_macro_exp')

        df_macro = df_monthly_macro.join(df_quarterly_macro, how='inner')
        df_macro = df_macro.add_suffix('_macro')
        log.debug(f"df macro features: {df_macro.keys()}")
        return df_macro, df_macro_exp
    except Exception as err:
        log.error(f"error getting macro data from db: {err}", exc_info=True)
        return None, None

def get_quarterly_data(start_date):
    try:
        client = finnhub.Client(api_key=config.finnhub.apikey)
        # create base df to add to later - starting with gnp
        gnp_dict = {
            'date': [],
            'gnp': [],
        }

        try:
            response = client.economic_data('MA-USA-717880')    
        except Exception:
            log.warning("error getting quarterly data - retrying")
            time.sleep(10)
            response = client.economic_data('MA-USA-717880')    

        data = response['data']
        for entry in data:
            date = datetime.strptime(entry.get('date'), "%Y-%m-%d")
            date = pytz.timezone('US/Eastern').localize(date) # TODO: verify this
            date = date.astimezone(pytz.timezone('UTC'))
            if date > start_date:
                gnp_dict['date'].append(date)
                gnp_dict['gnp'].append(entry.get('value'))
        df_macro = pd.DataFrame(gnp_dict)
        df_macro = df_macro.set_index('date').sort_index()
        
        # get remaining data and concat with final df  
        macro_data = {
            'gdp_annual_growth_rate': 'MA-USA-71657182',
            # 'gnp': 'MA-USA-717880',
            'cur_account': 'MA-USA-6765',
            'bankruptcies': 'MA-USA-6682',
            'corporate_profits': 'MA-USA-6780',
            'gdp_growth_rate': 'MA-USA-717182',
        }

        # for each macro key and code, get data, save as temp df, concat with final df
        for key, code in macro_data.items():
            code_dict = {
                'date': [],
                key: [],
            }

            try:
                response = client.economic_data(code)
            except Exception:
                log.warning("error getting quarterly data - retrying")
                response = client.economic_data(code)

            time.sleep(float(config.time_buffer))
            data = response['data']
            for entry in data:
                date = datetime.strptime(entry.get('date'), "%Y-%m-%d")
                date = pytz.timezone('US/Eastern').localize(date) # TODO: verify this
                date = date.astimezone(pytz.timezone('UTC'))
                if date > start_date:
                    code_dict['date'].append(date)
                    code_dict[key].append(entry.get('value'))
            df_temp = pd.DataFrame(code_dict)
            df_temp = df_temp.set_index('date').sort_index()

            # concat with final df
            df_macro = pd.concat([df_macro, df_temp], axis=1)

        # replace NaN's with None so it can be stored in db
        df_macro = df_macro.replace({np.nan: None})

        # original data
        for i in range(len(df_macro['gnp'])):
            ensure_connection() # ensure mysql connection
            QuarterlyMacroData(
                date=df_macro.index[i],

                gdp_growth_rate=df_macro['gdp_growth_rate'][i],
                gdp_annual_growth_rate=df_macro['gdp_annual_growth_rate'][i],
                gnp=df_macro['gnp'][i],
                cur_account=df_macro['cur_account'][i],
                bankruptcies=df_macro['bankruptcies'][i],
                corporate_profits=df_macro['corporate_profits'][i],
            ).save()
        return 
    except Exception as err:
        log.error(f"error getting quarterly data df: {err}", exc_info=True)
        return
    
def get_annual_data(start_date):
    try:
        client = finnhub.Client(api_key=config.finnhub.apikey)

        # create base df to add to later - starting with gdp
        gdp_dict = {
            'date': [],
            'gdp': [],
        }

        try:
            response = client.economic_data('MA-USA-71')    
        except Exception:
            log.warning("error getting annual macro data - retrying")
            time.sleep(10)
            response = client.economic_data('MA-USA-71')   
    
        data = response['data']
        for entry in data:
            date = datetime.strptime(entry.get('date'), "%Y-%m-%d")
            date = pytz.timezone('US/Eastern').localize(date) # TODO: verify this
            date = date.astimezone(pytz.timezone('UTC'))
            if date > start_date:
                gdp_dict['date'].append(date)
                gdp_dict['gdp'].append(entry.get('value'))
        df_macro = pd.DataFrame(gdp_dict)
        df_macro = df_macro.set_index('date').sort_index()
        
        # get remaining data and concat with final df  
        macro_data = {
            # 'gdp': 'MA-USA-71',
            'gdp_per_capita': 'MA-USA-718067',
            'cur_account_to_gdp': 'MA-USA-6765716880',
        }

        # for each macro key and code, get data, save as temp df, concat with final df
        for key, code in macro_data.items():
            code_dict = {
                'date': [],
                key: [],
            }

            try:
                response = client.economic_data(code)
            except Exception:
                log.warning("error getting annual macro data - retrying")
                time.sleep(10)
                response = client.economic_data(code)
                
            time.sleep(float(config.time_buffer))
            data = response['data']
            for entry in data:
                date = datetime.strptime(entry.get('date'), "%Y-%m-%d")
                date = pytz.timezone('US/Eastern').localize(date) # TODO: verify this
                date = date.astimezone(pytz.timezone('UTC'))
                if date > start_date:
                    code_dict['date'].append(date)
                    code_dict[key].append(entry.get('value'))
            df_temp = pd.DataFrame(code_dict)
            df_temp = df_temp.set_index('date').sort_index()

            # concat with final df
            df_macro = pd.concat([df_macro, df_temp], axis=1)

        # replace NaN's with None so it can be stored in db
        df_macro = df_macro.replace({np.nan: None})

        # original data
        for i in range(len(df_macro['gdp'])):
            ensure_connection() # ensure mysql connection
            AnnualMacroData(
                date=df_macro.index[i],

                gdp=df_macro['gdp'][i],
                gdp_per_capita=df_macro['gdp_per_capita'][i],
                cur_account_to_gdp=df_macro['cur_account_to_gdp'][i],
            ).save()
        return 
    except MySQLdb.OperationalError as err:
        log.error(err)
        log.error("restarting connection")
        get_annual_data(start_date=start_date)
    except Exception as err:
        log.error(f"error getting annual data: {err}", exc_info=True)
        return

def get_all_codes():
    try:
        client = finnhub.Client(api_key=config.finnhub.apikey)
        response = client.economic_code()
        for i in response:
            if i['country'] == 'United States': 
                log.debug(i)
    except Exception as err:
        log.error(f"error getting econ codes: {err}", exc_info=True)
    return

def get_code_by_name(name):
    try:
        client = finnhub.Client(api_key=config.finnhub.apikey)
        response = client.economic_code()
        for i in response:
            if i['name'] == name and i['country'] == 'United States':
                log.debug(i)
    except Exception as err:
        log.error(f"error: {err}", exc_info=True)
    return

def get_data_by_code(code, start_date):
    try:
        client = finnhub.Client(api_key=config.finnhub.apikey)
        response = client.economic_data(code)
        data = response['data']
        filtered_data = [x for x in data if datetime.strptime(x['date'], "%Y-%m-%d") > start_date]
        return filtered_data
    except Exception as err:
        log.error(f"error getting data for code: {code}: {err}", exc_info=True)
        return -1
    

def get_cpi():
    try:
        client = finnhub.Client(api_key=config.finnhub.apikey)
        start_date = config.start_date
        response = client.economic_data('MA-USA-678073')
        data = response["data"]

        for i in data:
            if datetime.strptime(i["date"], "%Y-%m-%d") > start_date:
                log.debug(i)
    except Exception as err:
        log.error(f"error: {err}", exc_info=True)
    return

def get_inflation():
    client = finnhub.Client(api_key=config.finnhub.apikey)
    response = client.economic_data('MA-USA-67807367')
    data = response["data"]
    for i in data:
        if datetime.strptime("2015-01-01", "%Y-%m-%d"):
            log.debug(i)
    return

def realTimeMacroData():
    try:
        log.debug(f"getting current macro data at {datetime.now().time()}")

        # clear db data and retrieve new data - lazy but doesnt take long and is simple
        MonthlyMacroData.objects.all().delete()
        QuarterlyMacroData.objects.all().delete()
        AnnualMacroData.objects.all().delete()

        # get new data
        get_macro(start_date=config.start_date)
    except Exception as err:
        log.error(f"error getting real time macro data: {err}", exc_info=True)
        

def get_macro_indicators(df):
    try:
        df_copy = df.copy()
        for key, values in df_copy.items():
            if (df_copy[key].isna().all()):
                log.info(f"get_macro_indicators: {key} data empty - skipping: {df_copy[key]}")
                continue

            sma4_key = key + '_sma4'
            df_copy[sma4_key] = ta.sma(df_copy[key], length=4)

            sma_diff_key = key + '_sma4_diff'
            df_copy[sma_diff_key] = df_copy[key] - df_copy[sma4_key]

            diff_key = key + '_diff'
            df_copy[diff_key] = df_copy[key].diff()

            qoq_key = key + "_qoq"
            df_copy[qoq_key] = df_copy[diff_key].diff()

            # cross above/below 
            prev_value = df_copy[key].shift(1)
            prev_sma4 = df_copy[sma4_key].shift(1)

            if (df_copy[sma4_key].isna().all()):
                log.info(f"sma4 {key} data empty - skipping: {df_copy[sma4_key]}")
                continue
            
            # if latest value just crossed above sma4: 1. otherwise, 0
            cross_above_key = key + '_cross_above'
            df_copy[cross_above_key] = ((df_copy[key] > df_copy[sma4_key]) & (prev_value <= prev_sma4)).astype(int)

            # if latest value just crossed below sma4: 1. otherwise, 0
            cross_below_key = key + '_cross_below'
            df_copy[cross_below_key] = ((df_copy[key] < df_copy[sma4_key]) & (prev_value >= prev_sma4)).astype(int)

            # diverging above/below
            df_copy['std4'] = ta.stdev(df_copy[key], length=4)

            diverge_above_key = key + '_dvg_above'
            df_copy[diverge_above_key] = (df_copy[key] > (df_copy[sma4_key] + df_copy['std4'])).astype(int)
            
            diverge_below_key = key + '_dvg_below'
            df_copy[diverge_below_key] = (df_copy[key] < (df_copy[sma4_key] - df_copy['std4'])).astype(int)

            df_copy.drop(columns=['std4'], inplace=True)
        return df_copy
    except Exception as err:
        log.error(f"error getting macro indicators: {err}", exc_info=True)
        return