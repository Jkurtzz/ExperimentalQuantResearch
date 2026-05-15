# ExpQuantResearch

An experimental quantitative trading research platform built with Django that investigates multiple alternative and traditional data sources to generate trading signals using Random Forest machine learning models. The system fetches data from Finnhub, performs sentiment analysis via OpenAI, and executes trades through the Alpaca paper trading API.

## Architecture

The project is structured as a Django application with a MySQL database backend. It follows a pipeline architecture:

```
Data Collection → Feature Engineering → Model Training → Signal Generation → Trade Execution
```

### Project Structure

```
src/
├── manage.py                          # Django entry point
├── resources/
│   └── config.yaml                    # Configuration (API keys, model params, indicators)
├── ExpQuantResearch/                  # Django project settings & management commands
│   ├── settings.py                    # Django config (MySQL, logging)
│   └── management/commands/
│       ├── startup.py                 # Full pipeline: data fetch → train → trade
│       ├── backtest.py                # Run backtesting simulation
│       ├── backtestMacro.py           # Backtest macro-based market timing
│       ├── backtestStockSelection.py  # Backtest stock selection strategy
│       ├── backtestSymbols.py         # Backtest symbol selection
│       ├── buy_stock.py               # Test trade execution connectivity
│       ├── continuousNews.py          # Continuous news ingestion for all US stocks
│       ├── merge_stock_data.py        # Multi-source feature combination & selection
│       └── update_stock_data.py       # Batch data update & CSV export
└── core/                              # Core application logic
    ├── models.py                      # Django ORM models (20+ tables)
    ├── config.py                      # YAML config loader (singleton)
    ├── control.py                     # Main orchestration (startup, model creation)
    ├── trainingUtils.py               # Random Forest training & feature selection
    ├── stratTestUtils.py              # Backtesting engine
    ├── symbolUtils.py                 # Stock discovery & market regime detection
    ├── realTimeUtils.py               # Real-time trading decision engine
    ├── realtime/intraday.py           # Alpaca WebSocket streaming (dev)
    ├── newsUtils.py                   # News data (Finnhub + Alpaca)
    ├── socialUtils.py                 # Social media sentiment (Finnhub)
    ├── pressUtils.py                  # Press release analysis
    ├── earningsUtils.py               # Quarterly earnings & transcripts
    ├── macroUtils.py                  # Macroeconomic indicators
    ├── insiderUtils.py                # Insider transaction analysis
    ├── insiderPressUtils.py           # Statistical hypothesis testing
    ├── sentimentUtils.py              # OpenAI-based sentiment scoring
    ├── dailyUtils.py                  # Daily OHLCV & pivot levels
    ├── intraDayUtils.py               # Intraday data & 20+ technical indicators
    ├── dbUtils.py                     # Database connection management
    └── utils.py                       # Data cleaning, normalization, visualization
```

## Data Sources

The system ingests six categories of data, all fetched primarily from the **Finnhub API**:

| Data Type | Source | Key Features |
|-----------|--------|-------------|
| **News** | Finnhub, Alpaca | Article sentiment (via OpenAI), sentiment volume, momentum |
| **Social Media** | Finnhub | Social sentiment scores, volume, sentiment-volume interaction |
| **Press Releases** | Finnhub | Sentiment, toneshift analysis, origin/schedule classification |
| **Earnings** | Finnhub | 40+ financial metrics (balance sheet, income, cash flow), transcript sentiment |
| **Macro** | Finnhub | 25+ monthly indicators (CPI, PMI, unemployment), quarterly GDP, annual data |
| **Insider Trading** | Finnhub | Transaction volumes, dollar volume change, spike detection, blackout periods |

**Price data** is sourced from **Alpaca** (hourly and daily bars) with 20+ technical indicators calculated including SMA, EMA, RSI, Bollinger Bands, MACD, Stochastic, OBV, ATR, CCI, VWAP, CMF, and Williams %R.

**Sentiment analysis** is performed via **OpenAI** (GPT-4o-mini / GPT-4o) for news articles, press releases, and earnings transcripts. Each item receives a sentiment score from -1 (bearish) to +1 (bullish), and press releases additionally receive toneshift, origin, and schedule classifications.

## Analysis Pipeline

### 1. Stock Selection

The system queries all US-listed stocks (NYSE, AMEX, BATS, NASDAQ) and filters candidates based on:
- Market cap, EPS, and ROE thresholds
- Beta ranges adjusted by market regime (bullish/neutral/bearish)
- RSI filters and dollar volume trends
- Insider transaction signals

Market regime is determined across three timeframes (short: 5 days, medium: 10 days, long: 20 days) using SPY trend analysis, SMA/EMA crosses, RSI, advance/decline ratios, and volume trends.

### 2. Feature Engineering

Raw data from each source undergoes a standardized processing pipeline:

1. **Indicator calculation** — Rolling window averages (short/medium/long), momentum, interactions, divergences
2. **Rate of change** — First and second derivatives of indicators
3. **Winsorization** — Outlier capping at configurable percentiles
4. **Z-score normalization** — Standardization for model input
5. **Exponential decay** — Gap-filling for sparse data (insider, press)

### 3. Model Training

For each stock symbol and indicator type, the system:

1. Generates binary trade signals based on forward-looking price changes (e.g., 1% move within a timeframe)
2. Tests **750 random feature combinations** across multiple timeframes using multi-threaded evaluation
3. Trains **Random Forest classifiers** (100 trees, 80/20 time-based split) on the best feature sets
4. Validates models against minimum precision and accuracy thresholds
5. Optionally trains a **safety model** (inverse prediction) to block false signals
6. Saves models as `.joblib` files

### 4. Signal Generation & Trade Execution

**Real-time mode:**
- Monitors indicators hourly (intraday) or daily
- Loads trained models and predicts on latest data
- Requires **ensemble consensus** (2+ indicators agreeing) before trading
- Safety model can block trades if it predicts the opposite direction
- Executes bracket orders via Alpaca with automated stop-loss (2/3 of target) and take-profit

**Backtesting mode:**
- Simulates weekly trading cycles with Friday retraining
- Tracks portfolio performance vs. SPY benchmark
- Entry at 11:30 AM, exit on stop-loss, take-profit, or timeframe expiration
- Logs win rates by trade type (long/short) and indicator

## Management Commands

```bash
# Full pipeline: discover stocks → fetch data → train models → start trading
python3 src/manage.py startup

# Run backtesting simulation
python3 src/manage.py backtest

# Backtest macro-based market timing strategy
python3 src/manage.py backtestMacro

# Backtest stock selection strategy
python3 src/manage.py backtestStockSelection

# Continuous news ingestion across all US stocks
python3 src/manage.py continuousNews

# Multi-source feature combination analysis
python3 src/manage.py merge_stock_data

# Batch data update and CSV export
python3 src/manage.py update_stock_data
```

## Configuration

All configuration lives in `src/resources/config.yaml`:

- **API credentials** — Alpaca, Finnhub, OpenAI, Polygon, Alpha Vantage
- **Model parameters** — Precision/accuracy thresholds, ensemble toggle, safety model toggle
- **Stock selection** — Market cap/EPS/ROE minimums, beta ranges per regime
- **Indicator windows** — SMA, EMA, RSI, Bollinger, MACD, Stochastic, ATR, CCI, CMF, WPR
- **Data source settings** — Rolling windows, winsorization levels, decay rates, timeframes per data type
- **Market regime thresholds** — Percentage change thresholds for bullish/bearish classification
- **Insider research** — Rolling window, percentile, minimum transaction counts

## Dependencies

- **Framework:** Django, MySQL
- **ML:** scikit-learn (RandomForestClassifier/Regressor), joblib
- **Data:** pandas, numpy
- **Technical Analysis:** pandas_ta
- **APIs:** finnhub-python, alpaca-trade-api, openai
- **Visualization:** plotly
- **Concurrency:** concurrent.futures (ThreadPoolExecutor)

## Database

MySQL database (`daytrade`) with 20+ tables covering:

- **Price data** — `StockMetrics`, `IntraDayData`, `DailyData`
- **Alternative data** — `FinnArticles`, `PressReleases`, `InsiderTransactions`, `SocialMedia`
- **Fundamentals** — `QEarnings`, `QEarningsSentiment`
- **Macro** — `MonthlyMacroData`, `QuarterlyMacroData`, `AnnualMacroData`
- **Stock selection** — `LongStocks`, `ShortStocks`, `ADRatios`
- **Trading results** — `TradeStats`, `BackTestTradeStats`
- **Research** — `InsiderPressResults`, `InsiderPressInstances`, `InsiderPressBlackout`, `InsiderPressStoch`

## Setup

1. Set up a MySQL database and update credentials in `src/resources/config.yaml`
2. Add API keys for Finnhub, Alpaca, and OpenAI to the config
3. Create a virtual environment and install dependencies
4. Run Django migrations:
   ```bash
   python3 src/manage.py migrate
   ```
5. Start the pipeline:
   ```bash
   python3 src/manage.py startup
   ```
