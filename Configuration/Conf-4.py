#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime as dt
import pytz
import logging
from Configuration import TradingPlatfomSettings as tps

platform = tps.platform

# INTC base params used as starting point for all securities (calibration will optimize)
_BASE_PARAMS = {
    'algorithm': 'MinerviniClaude',
    'entryByMarket': True,
    'exitTimeSeconds': 36000,
    'entryTimeSeconds': 3600,
    'minNumPastSamples': 51,
    'positionMargin': 0.0017149999999999997,
    'stopLossCoefficient': 25,
    'period': '1Min',
    # VCP
    'VCP_ATR_SLOPE_EXPANSION': 0.1620308857142857,
    'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
    'VCP_ADX_TREND_THRESHOLD': 13,
    # Indicator Periods
    'EMA_FAST': 14,
    'EMA_MID': 16,
    'EMA_SLOW': 24,
    'RSI_PERIOD': 23,
    'ATR_PERIOD': 19,
    'ATR_SLOPE_WINDOW': 5,
    'ADX_PERIOD': 13,
    'BB_WINDOW': 10,
    'BB_STD': 1,
    'BB_PERCENTILE_WINDOW': 49,
    'FVP_WINDOW': 47,
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
    'MARGIN_CONTRACTION_FIXED': 0.001365,
    'MARGIN_EXPANSION_MULTIPLIER': 1.2288571428571429,
    'MARGIN_EXPANSION_MIN': 0.00098,
    'MARGIN_EXPANSION_MAX': 0.00488,
    'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
    'MARGIN_TREND_MIN': 0.0033799999999999998,
    'MARGIN_TREND_MAX': 0.01014,
    # Calibration Parameters (3 months lookback for TEST_OFFLINE)
    'CALIBRATION_LOOKBACK_DAYS': 90,
    'CALIBRATION_LIMIT_RESULTS': 40000,
    'CALIBRATION_MIN_ROWS': 1000,
    'CALIBRATION_MARGIN_MIN': 0.001,
    'CALIBRATION_MARGIN_MAX': 0.006,
    'CALIBRATION_MARGIN_STEPS': 10,
    # Calibration Simulation Parameters
    'CALIBRATION_LOOKAHEAD_BARS': 60,
    'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
    'CALIBRATION_DEFAULT_MARGIN': 0.003,
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
    'MIN_CONFIDENCE': 0.294,
}

def _sec(code, decimals=2, timezone='America/New_York', currency='USD',
         exchange='SMART', primary_exchange=None,
         trading_times=(dt.time(9, 44), dt.time(15, 45)),
         time2close=dt.time(16, 30)):
    sec = {
        'seccode': code,
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': decimals,
        'id': 0,
        'timezone': timezone,
        'currency': currency,
        'exchange': exchange,
        'tradingTimes': trading_times,
        'time2close': time2close,
        'params': dict(_BASE_PARAMS),
    }
    if primary_exchange:
        sec['primaryExchange'] = primary_exchange
    return sec

def _sec_eu(code, decimals=2, market='XETRA', timezone='Europe/Berlin',
            currency='EUR', exchange='SMART', primary_exchange='IBIS',
            trading_times=(dt.time(9, 0), dt.time(17, 30)),
            time2close=dt.time(17, 25)):
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

securities = [
    _sec('AAPL'),
    _sec('INTC', decimals=3),
    _sec('NVDA'),
    _sec('SOFI'),
    _sec('MARA'),
    _sec('RIVN'),
    _sec('HOOD'),
    _sec('SMCI'),
    _sec('DKNG'),
    _sec('MSTR'),
    _sec('AMZN'),
    _sec('MSFT'),
    # European stocks
    _sec_eu('RHM'),
    _sec_eu('SBX'),
    _sec_eu('BBVA', market='BME', timezone='Europe/Madrid', primary_exchange='BM'),
    _sec_eu('SAN', market='BME', timezone='Europe/Madrid', primary_exchange='BM'),
    # Germany - XETRA
    _sec_eu('IFX'),      # Infineon Technologies - semiconductor, beta 1.83
    _sec_eu('DBK'),      # Deutsche Bank - banking, beta 1.46
    _sec_eu('ENR'),      # Siemens Energy - energy, beta 1.60-1.81
    # France - Euronext Paris
    _sec_eu('GLE', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),      # Societe Generale - banking, beta 1.39
    _sec_eu('STMPA', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),    # STMicroelectronics - semiconductor, beta 1.22
    # Italy - Borsa Italiana
    _sec_eu('UCG', market='BVME', timezone='Europe/Rome', primary_exchange='BVME'),     # UniCredit - banking, beta 1.28
    _sec_eu('STLAM', market='BVME', timezone='Europe/Rome', primary_exchange='BVME'),   # Stellantis - automotive, beta 1.56
    # UK - London Stock Exchange
    _sec_eu('BARC', market='LSE', timezone='Europe/London', currency='GBP',
            primary_exchange='LSE',
            trading_times=(dt.time(8, 0), dt.time(16, 30)),
            time2close=dt.time(16, 25)),  # Barclays - banking, beta 1.98
]

logLevel = logging.DEBUG
#logLevel = logging.INFO
MODE = 'TEST_OFFLINE' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
periods = ['1Min'] #periods = ['1Min','30Min']
numDaysHistCandles = 89

simulation_net_balance = 29000

current_tz = pytz.timezone('America/New_York')
# 3 months ago to now
since = current_tz.localize(dt.datetime.now() - dt.timedelta(days=numDaysHistCandles))
until = current_tz.localize(dt.datetime.now())
between_time = (
    current_tz.localize(dt.datetime.strptime('07:00', '%H:%M')).time(),
    current_tz.localize(dt.datetime.strptime('23:40', '%H:%M')).time()
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
factorPosition_Balance = 0.23
factorMargin_Position  = 0.0035
entryTimeSeconds = 3600
exitTimeSeconds = 36000
stopLossCoefficient = 20
correction = 0.0
spread = 0.0

openaikey = platform['secrets']['openaikey']
