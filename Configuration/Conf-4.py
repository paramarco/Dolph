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

def _sec(code, decimals=2):
    return {
        'seccode': code,
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': decimals,
        'id': 0,
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
]

logLevel = logging.DEBUG
#logLevel = logging.INFO
MODE = 'INIT_DB' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
periods = ['1Min'] #periods = ['1Min','30Min']

simulation_net_balance = 29000

current_tz = pytz.timezone('America/New_York')
# 4 months ago to now
since = current_tz.localize(dt.datetime(year=2025, month=10, day=19, hour=10, minute=0))
until = current_tz.localize(dt.datetime.now())
between_time = (
    current_tz.localize(dt.datetime.strptime('07:00', '%H:%M')).time(),
    current_tz.localize(dt.datetime.strptime('23:40', '%H:%M')).time()
)
tradingTimes = (dt.time(9, 44), dt.time(15, 45))

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
