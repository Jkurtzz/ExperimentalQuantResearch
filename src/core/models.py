from django.db import models

''' migrate model: 
    $python3 manage.py makemigrations
    $python3 manage.py migrate
    '''
# tracks data about stocks at each date - allows for quicker backtesting - should be daily index
class StockMetrics(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=10, blank=True)
    date = models.DateTimeField(null=True, blank=True)
    timeframe = models.CharField(max_length=10, blank=True)

    close = models.FloatField(null=True, blank=True)
    high = models.FloatField(null=True, blank=True)
    low = models.FloatField(null=True, blank=True)
    open = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)

class StockSector(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=10, blank=True)
    sector = models.CharField(max_length=30, blank=True)
    
class LongStocks(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=10)
    beta = models.FloatField(null=True, blank=True)
    market_cap = models.IntegerField(null=True, blank=True)
    eps = models.FloatField(null=True, blank=True)
    roe = models.FloatField(null=True, blank=True)

class ShortStocks(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=10)
    beta = models.FloatField(null=True, blank=True)
    market_cap = models.IntegerField(null=True, blank=True)
    eps = models.FloatField(null=True, blank=True)
    roe = models.FloatField(null=True, blank=True)

class IntraDayData(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=10)
    timestamp = models.DateTimeField(null=True, blank=True)

    # prices
    open = models.FloatField(null=True, blank=True)
    high = models.FloatField(null=True, blank=True)
    low = models.FloatField(null=True, blank=True)
    close = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    dollar_volume = models.FloatField(null=True, blank=True) 
    
class DailyData(models.Model): # verify data
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=10)
    date = models.DateField(null=True, blank=True)

    open = models.FloatField(null=True, blank=True)
    high = models.FloatField(null=True, blank=True)
    low = models.FloatField(null=True, blank=True)
    close = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    dollar_volume = models.FloatField(null=True, blank=True)

    open_roc = models.FloatField(null=True, blank=True)
    high_roc = models.FloatField(null=True, blank=True)
    low_roc = models.FloatField(null=True, blank=True)
    close_roc = models.FloatField(null=True, blank=True)
    volume_roc = models.FloatField(null=True, blank=True)
    dollar_volume_roc = models.FloatField(null=True, blank=True)

    open_acc = models.FloatField(null=True, blank=True)
    high_acc = models.FloatField(null=True, blank=True)
    low_acc = models.FloatField(null=True, blank=True)
    close_acc = models.FloatField(null=True, blank=True)
    volume_acc = models.FloatField(null=True, blank=True)
    dollar_volume_acc = models.FloatField(null=True, blank=True)

    # percent changes 
    open_pct = models.FloatField(null=True, blank=True)
    high_pct = models.FloatField(null=True, blank=True)
    low_pct = models.FloatField(null=True, blank=True)
    close_pct = models.FloatField(null=True, blank=True)
    volume_pct = models.FloatField(null=True, blank=True)
    dollar_volume_pct = models.FloatField(null=True, blank=True)

    open_roc_pct = models.FloatField(null=True, blank=True)
    high_roc_pct = models.FloatField(null=True, blank=True)
    low_roc_pct = models.FloatField(null=True, blank=True)
    close_roc_pct = models.FloatField(null=True, blank=True)
    volume_roc_pct = models.FloatField(null=True, blank=True)
    dollar_volume_roc_pct = models.FloatField(null=True, blank=True)

    open_acc_pct = models.FloatField(null=True, blank=True)
    high_acc_pct = models.FloatField(null=True, blank=True)
    low_acc_pct = models.FloatField(null=True, blank=True)
    close_acc_pct = models.FloatField(null=True, blank=True)
    volume_acc_pct = models.FloatField(null=True, blank=True)
    dollar_volume_acc_pct = models.FloatField(null=True, blank=True)


class FinnArticles(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(null=True, blank=True, max_length=10)
    date = models.DateTimeField(null=True, blank=True)
    headline = models.TextField(null=True, blank=True)
    summary = models.TextField(null=True, blank=True)
    source = models.TextField(null=True, blank=True)
    url = models.TextField(null=True, blank=True)
    sentiment = models.FloatField(null=True, blank=True)
    count = models.IntegerField(null=True, blank=True)
    

class PressReleases(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(null=True, blank=True, max_length=10)

    date = models.DateTimeField(null=True, blank=True)
    headline = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    url = models.TextField(null=True, blank=True)
    sentiment_score = models.FloatField(null=True, blank=True)
    toneshift_score = models.FloatField(null=True, blank=True)
    count = models.IntegerField(null=True, blank=True)
    party = models.CharField(max_length=10, null=True, blank=True)
    schedule = models.CharField(max_length=15, null=True, blank=True)


class InsiderTransactions(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(null=True, blank=True, max_length=10)

    filing_date = models.DateTimeField(null=True, blank=True)
    transaction_date = models.DateTimeField(null=True, blank=True)

    id = models.TextField(null=True, blank=True)
    volume_change = models.FloatField(null=True, blank=True)
    transaction_price = models.FloatField(null=True, blank=True)
    transaction_code = models.CharField(null=True, blank=True, max_length=20)
    dollar_volume_change = models.FloatField(null=True, blank=True)
    transaction_count = models.FloatField(null=True, blank=True)


class SocialMedia(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(null=True, blank=True, max_length=10)
    date = models.DateTimeField(null=True, blank=True)

    # standard data
    sentiment = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    sentiment_volume = models.FloatField(null=True, blank=True) # sentiment-volume score

class QEarnings(models.Model): 
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(null=True, blank=True, max_length=10)
    date = models.DateTimeField(null=True, blank=True)

    # balance sheet
    total_assets = models.FloatField(null=True, blank=True)
    total_liabilities = models.FloatField(null=True, blank=True)
    shareholders_equity = models.FloatField(null=True, blank=True)
    current_assets = models.FloatField(null=True, blank=True)
    current_liabilities = models.FloatField(null=True, blank=True)
    cash = models.FloatField(null=True, blank=True)

    # cash flow statement
    operating_cash_flow = models.FloatField(null=True, blank=True)
    investing_cash_flow = models.FloatField(null=True, blank=True)
    financing_cash_flow = models.FloatField(null=True, blank=True)
    depreciation_amortization = models.FloatField(null=True, blank=True)
    net_income = models.FloatField(null=True, blank=True)
    cap_ex = models.FloatField(null=True, blank=True)

    # income statement
    operating_expenses = models.FloatField(null=True, blank=True)
    operating_margin = models.FloatField(null=True, blank=True)
    net_margin = models.FloatField(null=True, blank=True)
    gross_profit = models.FloatField(null=True, blank=True)
    gross_margin = models.FloatField(null=True, blank=True)
    free_cash_flow = models.FloatField(null=True, blank=True)
    debt_equity_ratio = models.FloatField(null=True, blank=True)

    ebit = models.FloatField(null=True, blank=True)
    ebitda = models.FloatField(null=True, blank=True)
    revenue = models.FloatField(null=True, blank=True)
    eps = models.FloatField(null=True, blank=True)
    roa = models.FloatField(null=True, blank=True)
    roe = models.FloatField(null=True, blank=True)
    shares_outstanding = models.FloatField(null=True, blank=True)
    bvps = models.FloatField(null=True, blank=True)

    # additional
    net_debt = models.FloatField(null=True, blank=True)
    total_debt = models.FloatField(null=True, blank=True)
    long_term_debt = models.FloatField(null=True, blank=True)
    total_debt_to_equity = models.FloatField(null=True, blank=True)

    cur_ratio = models.FloatField(null=True, blank=True)
    quick_ratio = models.FloatField(null=True, blank=True)
    cash_ratio = models.FloatField(null=True, blank=True)
    total_ratio = models.FloatField(null=True, blank=True)

    pb = models.FloatField(null=True, blank=True)
    ev = models.FloatField(null=True, blank=True)

    commonStock = models.FloatField(null=True, blank=True)
    fcf = models.FloatField(null=True, blank=True)
    ebitPerShare = models.FloatField(null=True, blank=True)

class QEarningsSentiment(models.Model):
    uid = models.BigAutoField(primary_key=True)
    symbol = models.CharField(null=True, blank=True, max_length=10)

    id = models.CharField(null=True, blank=True, max_length=25)
    time = models.DateTimeField(null=True, blank=True)
    title = models.TextField(null=True, blank=True)
    transcript = models.TextField(null=True, blank=True)

    sentiment_score = models.FloatField(null=True, blank=True)
    toneshift_score = models.FloatField(null=True, blank=True)


class MonthlyMacroData(models.Model):
    uid = models.BigAutoField(primary_key=True)
    date = models.DateTimeField(null=True, blank=True)

    cpi = models.FloatField(null=True, blank=True)
    core_consumer_prices = models.FloatField(null=True, blank=True)
    inflation_rate = models.FloatField(null=True, blank=True)
    inflation_rate_mom = models.FloatField(null=True, blank=True)
    ism_pmi = models.FloatField(null=True, blank=True)
    chicago_pmi = models.FloatField(null=True, blank=True)
    non_manufacturing_pmi = models.FloatField(null=True, blank=True)
    industrial_production = models.FloatField(null=True, blank=True)
    industrial_production_mom = models.FloatField(null=True, blank=True)
    unemployment_rate = models.FloatField(null=True, blank=True)
    non_farm_payrolls = models.FloatField(null=True, blank=True)
    adp_employment_change = models.FloatField(null=True, blank=True)
    labor_force_participation_rate = models.FloatField(null=True, blank=True)
    avg_hourly_earnings = models.FloatField(null=True, blank=True)
    bal_of_trade = models.FloatField(null=True, blank=True) # check unit change
    exports = models.FloatField(null=True, blank=True) # check unit change
    imports = models.FloatField(null=True, blank=True) # check unit change
    consumer_sentiment = models.FloatField(null=True, blank=True)
    nfib_boi = models.FloatField(null=True, blank=True)
    ibd_eoi = models.FloatField(null=True, blank=True)
    existing_home_sales = models.FloatField(null=True, blank=True)
    new_home_sales = models.FloatField(null=True, blank=True)
    housing_starts = models.FloatField(null=True, blank=True)
    pending_home_sales = models.FloatField(null=True, blank=True)
    housing_price_index_mom_change = models.FloatField(null=True, blank=True)

    # not monthly in api response
    # TODO: re-implement 
    # initial_jobless_claims = models.FloatField(null=True, blank=True)
    # continuing_jobless_claims = models.FloatField(null=True, blank=True)
    # three_month_interbank_rate = models.FloatField(null=True, blank=True)
    # api_crude_oil_stock_change = models.FloatField(null=True, blank=True)
    # crude_oil_stock_change = models.FloatField(null=True, blank=True)
    # natural_gas_stock_change = models.FloatField(null=True, blank=True)
    # baker_hughes_crude_oil_rigs = models.FloatField(null=True, blank=True)


class QuarterlyMacroData(models.Model):
    uid = models.BigAutoField(primary_key=True)
    date = models.DateTimeField(null=True, blank=True)

    gdp_growth_rate = models.FloatField(null=True, blank=True)
    gdp_annual_growth_rate = models.FloatField(null=True, blank=True)
    gnp = models.FloatField(null=True, blank=True)
    cur_account = models.FloatField(null=True, blank=True) # check for unit change
    bankruptcies = models.FloatField(null=True, blank=True)
    corporate_profits = models.FloatField(null=True, blank=True)


class AnnualMacroData(models.Model):
    uid = models.BigAutoField(primary_key=True)
    date = models.DateTimeField(null=True, blank=True)

    gdp = models.FloatField(null=True, blank=True)
    gdp_per_capita = models.FloatField(null=True, blank=True)
    cur_account_to_gdp = models.FloatField(null=True, blank=True)


class TradeStats(models.Model):
    uid = models.BigAutoField(primary_key=True)
    
    date = models.DateTimeField(null=True, blank=True) 
    order_id = models.TextField(null=True, blank=True)
    take_profit_id = models.TextField(null=True, blank=True)
    stop_loss_id = models.TextField(null=True, blank=True)

    success = models.BooleanField(null=True, blank=True)
    indicator_type = models.CharField(max_length=15, null=True, blank=True) # still need to figure out
    pct_change = models.FloatField(null=True, blank=True)
    trade_type = models.CharField(max_length=10, blank=True)

class BackTestTradeStats(models.Model):
    uid = models.BigAutoField(primary_key=True)
    
    date = models.DateTimeField(null=True, blank=True) 

    symbol = models.CharField(max_length=10, null=True, blank=True)
    success = models.BooleanField(null=True, blank=True)
    indicator_type = models.CharField(max_length=15, null=True, blank=True) # still need to figure out
    pct_change = models.FloatField(null=True, blank=True)
    trade_type = models.CharField(max_length=10, blank=True)

class ADRatios(models.Model):
    uid = models.BigAutoField(primary_key=True)

    date = models.DateTimeField(null=True, blank=True)
    ad_ratio = models.FloatField(null=True, blank=True)

class InsiderPressResults(models.Model):
    uid = models.BigAutoField(primary_key=True)

    sector_combination = models.TextField(null=True, blank=True)
    start_date = models.DateField(null=True, blank=True)
    timeframe = models.IntegerField(null=True, blank=True)
    market_regime = models.TextField(null=True, blank=True)
    transaction_type = models.TextField(null=True, blank=True)
    market_cap_group = models.TextField(null=True, blank=True)
    beta_group = models.TextField(null=True, blank=True)

    percentile = models.FloatField(null=True, blank=True)
    min_transactions = models.IntegerField(null=True, blank=True)
    min_buys = models.IntegerField(null=True, blank=True)
    min_sells = models.IntegerField(null=True, blank=True)

    correct_spikes = models.IntegerField(null=True, blank=True)
    total_spikes = models.IntegerField(null=True, blank=True)
    min_z_score = models.FloatField(null=True, blank=True)
    median_z_score = models.FloatField(null=True, blank=True)
    
    observed_ratio = models.FloatField(null=True, blank=True)
    p_val = models.FloatField(null=True, blank=True)

    correct_short_term_dir_ratio = models.FloatField(null=True, blank=True)
    short_term_dir_p_val = models.FloatField(null=True, blank=True)
    correct_long_term_dir_ratio = models.FloatField(null=True, blank=True)
    long_term_dir_p_val = models.FloatField(null=True, blank=True)

class InsiderPressInstances(models.Model):
    uid = models.BigAutoField(primary_key=True)

    symbol = models.CharField(max_length=5, null=True, blank=True)
    sector = models.TextField(null=True, blank=True)
    date = models.DateTimeField(null=True, blank=True)

    market_regime = models.TextField(null=True, blank=True) # at time of transaction
    beta = models.FloatField(null=True, blank=True)
    market_cap = models.FloatField(null=True, blank=True)

    transaction_date = models.DateTimeField(null=True, blank=True) # actually filing date 
    transaction_type = models.CharField(max_length=9, null=True, blank=True)
    dv_change = models.FloatField(null=True, blank=True)
    dv_z_score = models.FloatField(null=True, blank=True)
    dv_percentile = models.FloatField(null=True, blank=True)
    
    press_occurred = models.BooleanField(null=True, blank=True)
    pre_price_change = models.FloatField(null=True, blank=True) # price change from insider spike to day of press release in excess return units

    press_date = models.DateTimeField(null=True, blank=True)
    press_schedule = models.TextField(null=True, blank=True) # scheduled vs unscheduled - should be first party only
    sentiment_score = models.FloatField(null=True, blank=True)
    toneshift_score = models.FloatField(null=True, blank=True)

    days_between = models.IntegerField(null=True, blank=True)

    post_press_movements = models.JSONField(null=True, blank=True) # return after press
    no_press_movements = models.JSONField(null=True, blank=True) # return if no press

class InsiderPressBlackout(models.Model):
    uid = models.BigAutoField(primary_key=True)

    symbol = models.CharField(max_length=5, null=True, blank=True)
    sector = models.TextField(null=True, blank=True)
    date = models.DateTimeField(null=True, blank=True)
    transaction_type = models.CharField(max_length=9, null=True, blank=True)

    days_since_last_transaction = models.IntegerField(null=True, blank=True)
    q95_blackout_period = models.FloatField(null=True, blank=True)

    press_date = models.DateTimeField(null=True, blank=True)
    press_schedule = models.TextField(null=True, blank=True)


class InsiderPressStoch(models.Model):
    uid = models.BigAutoField(primary_key=True)

    symbol = models.CharField(max_length=5, null=True, blank=True)
    firm_id = models.IntegerField(null=True, blank=True)
    sector = models.TextField(null=True, blank=True)
    date = models.DateTimeField(null=True, blank=True) # date or datetime?

    dv_change = models.FloatField(null=True, blank=True)
    dv_z_score = models.FloatField(null=True, blank=True)

    # press
    press_occurred = models.IntegerField(null=True, blank=True)
    press_schedule = models.TextField(null=True, blank=True)

    # controls 
    market_cap = models.FloatField(null=True, blank=True)
    atr = models.FloatField(null=True, blank=True)
    earnings_window = models.IntegerField(null=True, blank=True)
    market_regime = models.IntegerField(null=True, blank=True)