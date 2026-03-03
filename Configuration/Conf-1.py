#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime as dt
import pytz
import logging
from Configuration import TradingPlatfomSettings as tps

platform = tps.platform

_BASE_PARAMS = {
    'algorithm': 'MinerviniClaude',
    'entryByMarket': False,
    'exitTimeSeconds': 7200,
    'entryTimeSeconds': 360,
    'minNumPastSamples': 51,
    'positionMargin': 0.003,
    'stopLossCoefficient': 8,
    'period': '1Min',
    # ===== Phase Detection Parameters =====
    # Expansion:  (ATR_slope > threshold AND Bollinger width percentile high)
    # Trend: ( ADX above threshold, strong directional movement) and (EMA alignment either bullish or bearish)
    'VCP_ATR_SLOPE_EXPANSION': 0.1620308857142857,
    'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
    'VCP_ADX_TREND_THRESHOLD': 13,
    # Indicator Periods
    'EMA_FAST': 14,
    'EMA_MID': 16,
    'EMA_SLOW': 24,
    'RSI_PERIOD': 23,       # RSI (Momentum Filter). to Confirm directional entries, Filter false breakouts
    'ATR_PERIOD': 19,       # ATR CALCULATION. to measure volatility level
    'ATR_SLOPE_WINDOW': 5,  # ATR SLOPE. Measures volatility expansion / contraction speed
    'ADX_PERIOD': 13,       # ADX + DI (Trend Strength). to confirm directional strength, to filter ranging markets
    'BB_WINDOW': 10,        # BOLLINGER BAND WIDTH. Measures compression vs expansion of volatility
    'BB_STD': 1,
    'BB_PERCENTILE_WINDOW': 49,
    'FVP_WINDOW': 47,       # FAIR VALUE PRICE (FVP). Rolling mean of close. Used for mean-reversion during expansion
    # Expansion Phase Thresholds
    'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
    'EXPANSION_RSI_SHORT_MIN': 20,
    'EXPANSION_RSI_LONG_MAX': 50,
    # Trend Phase Thresholds
    'TREND_RSI_LONG_MIN': 20,
    'TREND_RSI_LONG_MAX': 95,
    'TREND_RSI_SHORT_MIN': 15,
    'TREND_RSI_SHORT_MAX': 60,
    # Margin Adaptation Parameters
    'MARGIN_CONTRACTION_FIXED': 0.0015,
    'MARGIN_EXPANSION_MULTIPLIER': 1.5,
    'MARGIN_EXPANSION_MIN': 0.004,
    'MARGIN_EXPANSION_MAX': 0.015,
    'MARGIN_TREND_ATR_MULTIPLIER': 2.0,
    'MARGIN_TREND_MIN': 0.003,
    'MARGIN_TREND_MAX': 0.010,
    # Calibration Parameters (3 months lookback for mode = TEST_OFFLINE)
    'CALIBRATION_LOOKBACK_DAYS': 90,
    'CALIBRATION_LIMIT_RESULTS': 40000,
    'CALIBRATION_MIN_ROWS': 1000,
    'CALIBRATION_MARGIN_MIN': 0.001,
    'CALIBRATION_MARGIN_MAX': 0.015,
    'CALIBRATION_MARGIN_STEPS': 10,
    # Calibration Simulation Parameters
    'CALIBRATION_LOOKAHEAD_BARS': 60,
    # Volume Analysis Parameters
    'VOLUME_AVG_WINDOW': 13,
    'VOLUME_SLOPE_WINDOW': 4,
    'BIG_VOLUME_THRESHOLD': 1.638,
    'EXTREME_VOLUME_THRESHOLD': 2.7001,
    'BIG_BODY_ATR_THRESHOLD': 0.588,
    'EXTREME_BODY_ATR_THRESHOLD': 3.157142857142857,
    'DIVERGENCE_LOOKBACK': 10,
    # Buying Climax
    'BUYING_CLIMAX_LOOKBACK': 10,
    'BUYING_CLIMAX_TREND_LOOKBACK': 7,
    'BUYING_CLIMAX_EXTENSION': 0.005481285714285714,
    'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
    # Final Decision Scoring
    'MIN_TOTAL_SCORE': 0.735,
    'MIN_CONFIDENCE': 0.8,
    # Volume Confirmation Gate (Idea #3)
    'MIN_RELATIVE_VOLUME': 0.8,
    # Position Management
    'POSITION_COOLDOWN_SECONDS': 300,
    # Confidence Penalties
    'NO_VOLUME_CONFIDENCE_PENALTY': 0.40,
    'MOMENTUM_LOOKBACK': 5,
    'COUNTER_TREND_THRESHOLD': 0.003,
    'COUNTER_TREND_FACTOR': 10.0,
}

def _sec(code, decimals=2, timezone='America/New_York', currency='USD',
         exchange='SMART', primary_exchange='NASDAQ',
         trading_times=(dt.time(9, 46), dt.time(15, 45)),
         time2close=dt.time(15, 53)):
    return {
        'seccode': code,
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': decimals,
        'id': 0,
        'timezone': timezone,
        'currency': currency,
        'exchange': exchange,
        'primaryExchange': primary_exchange,
        'tradingTimes': trading_times,
        'time2close': time2close,
        'params': dict(_BASE_PARAMS),
    }


def _sec_eu(code, decimals=2, market='XETRA', timezone='Europe/Berlin',
            currency='EUR', exchange='SMART', primary_exchange='IBIS',
            trading_times=(dt.time(9, 16), dt.time(15, 30)),
            time2close=dt.time(15, 38)):
    return {
        'seccode': code,
        'board': 'EQTY',
        'market': market,
        'decimals': decimals,
        'id': 0,
        'timezone': timezone,
        'currency': currency,
        'exchange': exchange,
        'primaryExchange': primary_exchange,
        'tradingTimes': trading_times,
        'time2close': time2close,
        'params': dict(_BASE_PARAMS),
    }

# ---------------------------------------------------------------------------
# Japan (TSE) helper
# Trading hours: 9:00-11:30 (morning) + 12:30-15:30 (afternoon), lunch break
# IB: exchange SMART, primaryExchange TSEJ, currency JPY
# ---------------------------------------------------------------------------
def _sec_jp(code, decimals=0,
            trading_times=(dt.time(9, 5), dt.time(15, 20)),
            time2close=dt.time(15, 25)):
    return {
        'seccode': code,
        'board': 'EQTY',
        'market': 'TSEJ',
        'decimals': decimals,
        'id': 0,
        'timezone': 'Asia/Tokyo',
        'currency': 'JPY',
        'exchange': 'SMART',
        'primaryExchange': 'TSEJ',
        'tradingTimes': trading_times,
        'time2close': time2close,
        'params': dict(_BASE_PARAMS),
    }

# ---------------------------------------------------------------------------
# Hong Kong (HKEX) helper
# Trading hours: 9:30-12:00 (morning) + 13:00-16:00 (afternoon), lunch break
# IB: exchange SMART, primaryExchange SEHK, currency HKD
# ---------------------------------------------------------------------------
def _sec_hk(code, decimals=2,
            trading_times=(dt.time(9, 35), dt.time(15, 55)),
            time2close=dt.time(15, 58)):
    return {
        'seccode': code,
        'board': 'EQTY',
        'market': 'SEHK',
        'decimals': decimals,
        'id': 0,
        'timezone': 'Asia/Hong_Kong',
        'currency': 'HKD',
        'exchange': 'SMART',
        'primaryExchange': 'SEHK',
        'tradingTimes': trading_times,
        'time2close': time2close,
        'params': dict(_BASE_PARAMS),
    }

securities = [
    # ==================== AMERICAS - NASDAQ  ====================
    _sec('PENN'),                 # Penn Entertain.- sports betting (ESPN Bet), beta ~2.0, range 3-5%
    _sec('AFRM'),                 # Affirm         - fintech/BNPL, high beta ~2.5, range 3-5%
    _sec('PLTR'),                 # Palantir       - AI/data analytics, high beta ~2.5, range 3-5%
    _sec('SHOP', primary_exchange='NYSE'),  # Shopify        - e-commerce platform, beta ~2.0, range 2-4%
    _sec('HOOD'),                 # Robinhood      - fintech/trading, volatile, range 3-5%
    _sec('SMCI'),                 # Super Micro    - AI servers, extreme volatility, range 4-8%
    _sec('DKNG'),                 # DraftKings     - sports betting, volatile, range 2-4%
    _sec('MSTR'),                 # MicroStrategy  - bitcoin treasury, extreme volatility, range 4-8%
    _sec('AMZN'),                 # Amazon         - e-commerce/cloud, liquid, range 1.5-2.5%
    #_sec('MSFT'),                 # Microsoft      - tech megacap, liquid, range 1-2%
    _sec('TSLA'),                 # Tesla          - EV/energy, very volatile, range 2-4%
    _sec('AMD'),                  # AMD            - semiconductor, high beta ~1.7, range 2-3%
    #_sec('AAPL'),                 # Apple          - tech megacap, liquid, range 1-2%
    _sec('INTC', decimals=3),     # Intel          - semiconductor, moderate volatility, range 1.5-2.5%
    _sec('NVDA'),                 # NVIDIA         - AI/GPU, very volatile, range 2-4%
    _sec('SOFI'),                 # SoFi           - fintech, high beta ~1.8, range 3-5%
    _sec('MARA'),                 # MARA Holdings  - bitcoin mining, extreme volatility, range 4-8%
    _sec('RIVN'),                 # Rivian         - EV startup, volatile, range 3-5%
    # ==================== EUROPE (12 securities) ====================
    # # Germany - XETRA
    # _sec_eu('RHM'),             # Rheinmetall    - defense, high beta ~1.5, range 2-3%
    # _sec_eu('SBX'),             # Stabilus       - industrial, moderate volatility, range 1.5-2.5%
    # _sec_eu('IFX'),             # Infineon       - semiconductor, beta 1.83, range 2-3%
    # _sec_eu('DBK'),             # Deutsche Bank  - banking, beta 1.46, range 1.5-2.5%
    # _sec_eu('ENR'),             # Siemens Energy - energy, beta 1.60-1.81, range 2-3%
    # # Spain - BME
    # _sec_eu('BBVA', market='BME', timezone='Europe/Madrid', primary_exchange='BM'),    # BBVA           - banking, beta 1.25, range 1.5-2.5%
    # _sec_eu('SAN', market='BME', timezone='Europe/Madrid', primary_exchange='BM'),     # Santander      - banking, beta 1.20, range 1.5-2.5%
    # # France - Euronext Paris
    # _sec_eu('GLE', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),     # Societe Gen.   - banking, beta 1.39, range 2-3%
    # _sec_eu('STMPA', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),   # STMicro        - semiconductor, beta 1.22, range 2-3%
    # # Italy - Borsa Italiana
    # _sec_eu('UCG', market='BVME', timezone='Europe/Rome', primary_exchange='BVME'),    # UniCredit      - banking, beta 1.28, range 2-3%
    # _sec_eu('STLAM', market='BVME', timezone='Europe/Rome', primary_exchange='BVME'),  # Stellantis     - automotive, beta 1.56 (FIXME: IB contract not found)
    # # UK - London Stock Exchange
    # _sec_eu('BARC', market='LSE', timezone='Europe/London', currency='GBP',
    #         primary_exchange='LSE',
    #         trading_times=(dt.time(9, 46), dt.time(14, 40)),
    #         time2close=dt.time(14, 45)),  # Barclays       - banking, beta 1.98, range 2-3%
    # # Netherlands - Euronext Amsterdam
    # _sec_eu('ASML', market='AEB', timezone='Europe/Amsterdam', primary_exchange='AEB'),  # ASML Holding   - semiconductor equip, beta ~1.3, range 2-3%
    # # France - Euronext Paris (new)
    # _sec_eu('BNP', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),       # BNP Paribas    - banking, beta ~1.3, range 1.5-2.5%
    # # Germany - XETRA (new)
    # _sec_eu('CBK'),             # Commerzbank    - banking, beta ~1.4, range 2-3%
    # # UK - London Stock Exchange (new)
    # _sec_eu('FLTR', market='LSE', timezone='Europe/London', currency='GBP',
    #         primary_exchange='LSE',
    #         trading_times=(dt.time(9, 46), dt.time(14, 40)),
    #         time2close=dt.time(14, 45)),  # Flutter Entert. - sports betting, beta ~1.3, range 2-3%
    # ==================== JAPAN - TSE (6 securities) ====================
    # _sec_jp('9984'),             # SoftBank Group  - tech/investment, beta ~1.5, range 2-3%
    # _sec_jp('8035'),             # Tokyo Electron  - semiconductor equip, very volatile, range 2-4%
    # _sec_jp('6857'),             # Advantest       - semiconductor test, high volatility, range 2-4%
    # _sec_jp('6920'),             # Lasertec        - semiconductor inspect, extreme volatility, range 3-5%
    # _sec_jp('9983'),             # Fast Retailing  - Uniqlo, heavy Nikkei weight, range 1.5-3%
    # _sec_jp('6758'),             # Sony Group      - tech/entertainment, liquid, range 1.5-2.5%
    # ==================== HONG KONG - HKEX (6 securities) ====================
    # _sec_hk('9988'),             # Alibaba Group   - e-commerce/cloud, very volatile, range 2-4%
    # _sec_hk('700'),              # Tencent         - tech/gaming, most liquid HKEX, range 1.5-3%
    # _sec_hk('3690'),             # Meituan         - delivery/tech, volatile, range 2-4%
    # _sec_hk('9618'),             # JD.com          - e-commerce, volatile, range 2-4%
    # _sec_hk('1810'),             # Xiaomi          - electronics/EV, high retail volume, range 2-3%
    # _sec_hk('1211'),             # BYD             - EV/batteries, volatile, range 2-3%
]

logLevel = logging.DEBUG
#logLevel = logging.INFO
MODE = 'TEST_OFFLINE' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
periods = ['1Min'] #periods = ['1Min','30Min']
numDaysHistCandles = 89
calibrationPauseSeconds = 3600  # 1 hour
simulation_net_balance = 29000
_tz = pytz.timezone('America/New_York')
since = _tz.localize(dt.datetime.now() - dt.timedelta(days=numDaysHistCandles))
until = _tz.localize(dt.datetime.now())
between_time = (
    _tz.localize(dt.datetime.strptime('07:00', '%H:%M')).time(),
    _tz.localize(dt.datetime.strptime('23:40', '%H:%M')).time()
)
tradingTimes = (dt.time(9, 46), dt.time(15, 45))
numTestSample = 500
TrainingHour = 10  # 10:00
currentTestIndex = 0

db_connection_params = {
    "dbname" : "dolph_db",
    "user" : "dolph_user",
    "password" : "dolph_password",
    "host" : "127.0.0.1",
    "port" : 4713,
    "sslmode" : "disable"
}

transaqConnectorPort = 13000
transaqConnectorHost = '127.0.0.1'

statusOrderForwarding = ['PendingSubmit', 'Submitted','PendingSubmit',  'watching', 'active', 'forwarding', 'new', 'pending_new', 'accepted', 'tp_guardtime', 'tp_forwarding', 'sl_forwarding', 'sl_guardtime','submitted', 'PreSubmitted', 'inactive']
statusOrderExecuted = ['Filled','tp_executed', 'sl_executed','filled','matched']
statusOrderCanceled = ['Cancelled','cancelled','Rejected','Stopped','denied', 'disabled', 'expired', 'failed', 'rejected', 'canceled', 'removed', 'done_for_day']
statusOrderOthers = ['PartiallyFilled','Inactive','PendingCancel', "linkwait","tp_correction","tp_correction_guardtime","none","inactive","wait","disabled","failed","refused","pending_cancel", "pending_replace", "stopped", "suspended", "calculated" ]
statusExitOrderExecuted = ['tp_executed', 'sl_executed','matched','triggered']
statusExitOrderFilled = ['filled','Filled']

########### default-fallback values ##########################################
factorPosition_Balance = 0.18    # DolphRobot.py:436, MinerviniClaude.py:956 (direct cm.X usage)

openaikey = platform['secrets']['openaikey']

# ===================================
# Calibration Simulation Parameters
# Used via getattr(cm, 'KEY', default) in MinerviniClaude._simulate_profit()
# ===================================
OPTIMAL_TP_RATIO_MIN = 0.0033
OPTIMAL_TP_RATIO_MAX = 0.0043
EXPIRED_PENALTY_FACTOR = 0.5
FREQ_TARGET_MIN = 3.0
FREQ_TARGET_MAX = 12.0
FREQ_SIGNAL_CONVERSION = 0.04

# ===================================
# Ideas #2-#6 Parameters
# Used via getattr(cm, 'KEY', default) in MinerviniClaude.py
# ===================================
# Idea #2: Multi-pass coordinate descent
MAX_CALIBRATION_PASSES = 2
MIN_CALIBRATION_IMPROVEMENT = 0.01
# Idea #3: Volume confirmation gate (module-level fallback)
MIN_RELATIVE_VOLUME = 0.8
# Idea #4: Margin dynamic cost floor
MIN_ABS_MARGIN_MULTIPLIER = 1.5
# Idea #5: Dynamic signal threshold by phase
EXPANSION_SCORE_MULT = 1.2
TREND_SCORE_MULT = 0.9
# Idea #6: Trailing TP in simulation
TRAILING_TP_ENABLED = True
TRAILING_TP_RETRACE = 0.50
# SL aversion: proportional penalty on real SL losses during calibration
CALIBRATION_SL_AVERSION = 0.15
