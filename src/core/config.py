import yaml
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, '../resources/config.yaml')

class Config:
    _instance = None

    # creates single instance of config manager
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
            self.source = config['source']
            self.symbol = config['symbol']
            self.start_date = config['start_date']
            self.cutoff_date = config['cutoff_date']
            self.time_buffer = config['time_buffer']
            self.tradingFrame = config['tradingFrame']
            self.arman = self.Arman(config['arman'])
            self.models = self.Models(config['models'])
            self.stocks = self.Stocks(config['stocks'])
            self.insiderPressResearch = self.InsiderPressResearch(config['insiderPressResearch'])
            self.database = self.Database(config['database'])
            self.alpaca = self.Alpaca(config['alpaca'])
            self.finnhub = self.Finnhub(config['finnhub'])
            self.poly = self.Polygon(config['polygon'])
            self.alpha = self.Alpha(config['alpha'])
            self.sma = self.SMA(config['sma'])
            self.ema = self.EMA(config['ema'])    
            self.rsi = self.RSI(config['rsi'])
            self.bollinger = self.Bollinger(config['bollinger'])
            self.macd = self.MACD(config['macd'])
            self.stoch = self.Stoch(config['stoch'])
            self.atr = self.ATR(config['atr'])
            self.cci = self.CCI(config['cci'])
            self.cmf = self.CMF(config['cmf'])
            self.wpr = self.WPR(config['wpr'])
    
    class Arman:
        def __init__(self, config):
            self.email = config['email']
            self.password = config['password']
    
    class InsiderPressResearch: 
        def __init__(self, config):
            self.rolling = config['rolling']
            self.timeframe = config['timeframe']
            self.percentile = config['percentile']
            self.min_transactions = config['min_transactions']
            self.min_buys = config['min_buys']
            self.min_sells = config['min_sells']

    class Models:
        def __init__(self, config):
            self.minPrecision = config['minPrecision']
            self.maxPrecision = config['maxPrecision']
            self.minAccuracy = config['minAccuracy']
            self.ensembleEnabled = config['ensembleEnabled']
            self.safetyEnabled = config['safetyEnabled']

    class Stocks:
        def __init__(self, config):
            self.numStocks = config['numStocks']
            self.minCap = config['minCap']
            self.minEps = config['minEps']
            self.minRoe = config['minRoe']
            self.short_term_market = self.MarketRange(config['short_term_market'])
            self.med_term_market = self.MarketRange(config['med_term_market'])
            self.long_term_market = self.MarketRange(config['long_term_market'])
            self.selection = self.MarketStatus(config['selection'])

        class MarketRange:
            def __init__(self, config):
                self.timeframe = config['timeframe']
                self.pos_pct_change = config['pos_pct_change']
                self.neg_pct_change = config['neg_pct_change']

        class MarketStatus:
            def __init__(self, config):
                self.bullish = self.StockSelection(config['bullish'])
                self.bearish = self.StockSelection(config['bearish'])

            class StockSelection:
                def __init__(self, config):
                    self.min_beta = config['min_beta']
                    self.max_beta = config['max_beta']
                    self.long_atr = config['long_atr']
                    self.short_atr = config['short_atr']
    
    class Database:
        def __init__(self, config):
            self.name = config['name']
            self.host = config['host']
            self.port = config['port']
            self.user = config['user']
            self.password = config['password']
            self.max_age = config['max_age']
    
    class Alpaca:
        def __init__(self, config):
            self.baseurl = config['baseurl']
            self.newsUrl = config['newsUrl']
            self.stockSocketUrl = config['stockSocketUrl']
            self.tradeUrl = config['tradeUrl']
            self.marketUrl = config['marketUrl']
            self.apikey = config['apikey']
            self.secret = config['secret']
            self.timeframe = config['timeframe']
            self.ext_hours = config['ext_hours']
            self.intra_day = self.IntraDay(config['intra_day'])
            self.daily = self.Daily(config['daily'])
        
        class IntraDay:
            def __init__(self, config):
                self.winsor_level = config['winsor_level']

        class Daily:
            def __init__(self, config):
                self.winsor_level = config['winsor_level']

    class Finnhub:
        def __init__(self, config):
            self.apikey = config['apikey']
            self.news = self.News(config['news'])
            self.press = self.Press(config['press'])
            self.insider = self.Insider(config['insider'])
            self.social = self.Social(config['social'])
            self.earnings = self.Earnings(config['earnings'])
            self.macro = self.Macro(config['macro'])

        class News:
            def __init__(self, config):
                self.summary_only = config['summary_only']
                self.latency = config['latency']
                self.decay = config['decay']
                self.winsor_level = config['winsor_level']
                self.long_window = config['long_window']
                self.medium_window = config['medium_window']
                self.short_window = config['short_window']
                self.timeframes = config['timeframes']

        class Press:
            def __init__(self, config):
                self.decay = config['decay']
                self.winsor_level = config['winsor_level']
                self.long_window = config['long_window']
                self.medium_window = config['medium_window']
                self.short_window = config['short_window']
                self.timeframes = config['timeframes']

        class Insider:
            def __init__(self, config):
                self.decay = config['decay']
                self.winsor_level = config['winsor_level']
                self.long_window = config['long_window']
                self.medium_window = config['medium_window']
                self.short_window = config['short_window']
                self.timeframes = config['timeframes']
        
        class Social:
            def __init__(self, config):
                self.long_window = config['long_window']
                self.medium_window = config['medium_window']
                self.short_window = config['short_window']
                self.winsor_level = config['winsor_level']
                self.timeframes = config['timeframes']

        class Earnings:
            def __init__(self, config):
                self.decay = config['decay']
                self.winsor_level = config['winsor_level']
                self.timeframes = config['timeframes']

        class Macro:
            def __init__(self, config):
                self.decay = config['decay']
                self.winsor_level = config['winsor_level']
                self.timeframes = config['timeframes']

    class Polygon:
        def __init__(self, config):
            self.baseurl = config['baseurl']
            self.apikey = config['apikey']
            self.adjusted = config['adjusted']
            self.multiplier = config['multiplier']
            self.timespan = config['timespan']
            self.limit = config['limit']
            self.sort = config['sort']
            self.ext_hours = config['ext_hours']
    
    class Alpha:
        def __init__ (self, config):
            self.baseUrl = config['baseUrl']
            self.apikey = config['apikey']
            self.function = config['function']
            self.interval = config['interval']
            self.ext_hours = config['ext_hours']
            self.output_size = config['output_size']

    class SMA:
        def __init__(self, config):
            self.low_window = config['low_window']
            self.mid_window = config['mid_window']
            self.high_window = config['high_window']
            
    class EMA:
        def __init__(self, config):
            self.low_window = config['low_window']
            self.mid_window = config['mid_window']
            self.high_window = config['high_window']
    
    class RSI:
        def __init__(self, config):
            self.low_window = config['low_window']
            self.high_window = config['high_window']

    class Bollinger:
        def __init__(self, config):
            self.window = config['window']
            self.std_dev = config['std_dev']

    class MACD:
        def __init__(self, config):
            self.low = self.Low(config['low'])
            self.high = self.High(config['high'])

        class Low:
            def __init__(self, config):
                self.short_window = config['short_window']
                self.long_window = config['long_window']
                self.signal_window = config['signal_window']
        
        class High:
            def __init__(self, config):
                self.short_window = config['short_window']
                self.long_window = config['long_window']
                self.signal_window = config['signal_window']
    
    class Stoch:
        def __init__(self, config):
            self.low = self.Low(config['low'])
            self.high = self.High(config['high'])

        class Low:
            def __init__(self, config):
                self.k = config['k']
                self.d = config['d']

        class High:
            def __init__(self, config):
                self.k = config['k']
                self.d = config['d']
    
    class ATR:
        def __init__(self, config):
            self.window = config['window']
    
    class CCI:
        def __init__(self, config):
            self.window = config['window']

    class CMF: 
        def __init__(self, config):
            self.window = config['window']

    class WPR:
        def __init__(self, config):
            self.window = config['window']
        

config = Config()
