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
    'entryByMarket': False,
    'exitTimeSeconds': 11400,
    'entryTimeSeconds': 3600,
    'minNumPastSamples': 51,
    'positionMargin': 0.003,
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
    'MARGIN_CONTRACTION_FIXED': 0.0015,
    'MARGIN_EXPANSION_MULTIPLIER': 1.5,
    'MARGIN_EXPANSION_MIN': 0.002,
    'MARGIN_EXPANSION_MAX': 0.008,
    'MARGIN_TREND_ATR_MULTIPLIER': 2.0,
    'MARGIN_TREND_MIN': 0.002,
    'MARGIN_TREND_MAX': 0.006,
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
    # Position Management
    'POSITION_COOLDOWN_SECONDS': 300,
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
    # --- US & EU temporarily disabled for Asia-only INIT_DB ---
    # _sec('TSLA'),
    # _sec('INTC', decimals=3),
    # _sec('AMD'),
    # _sec('SOFI'),
    # _sec('MARA'),
    # _sec('RIVN'),
    # _sec('HOOD'),
    # _sec('SMCI'),
    # _sec('DKNG'),
    # _sec('MSTR'),
    # _sec('AMZN'),
    # _sec('MSFT'),
    # # European stocks
    # _sec_eu('RHM'),
    # _sec_eu('SBX'),
    # _sec_eu('BBVA', market='BME', timezone='Europe/Madrid', primary_exchange='BM'),
    # _sec_eu('SAN', market='BME', timezone='Europe/Madrid', primary_exchange='BM'),
    # # Germany - XETRA
    # _sec_eu('IFX'),      # Infineon Technologies - semiconductor, beta 1.83
    # _sec_eu('DBK'),      # Deutsche Bank - banking, beta 1.46
    # _sec_eu('ENR'),      # Siemens Energy - energy, beta 1.60-1.81
    # # France - Euronext Paris
    # _sec_eu('GLE', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),      # Societe Generale - banking, beta 1.39
    # _sec_eu('STMPA', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),    # STMicroelectronics - semiconductor, beta 1.22
    # # Italy - Borsa Italiana
    # _sec_eu('UCG', market='BVME', timezone='Europe/Rome', primary_exchange='BVME'),     # UniCredit - banking, beta 1.28
    # _sec_eu('STLAM', market='BVME', timezone='Europe/Rome', primary_exchange='BVME'),   # Stellantis - automotive, beta 1.56
    # # UK - London Stock Exchange
    # _sec_eu('BARC', market='LSE', timezone='Europe/London', currency='GBP',
    #         primary_exchange='LSE',
    #         trading_times=(dt.time(8, 0), dt.time(16, 30)),
    #         time2close=dt.time(16, 25)),  # Barclays - banking, beta 1.98
    # ==================== JAPAN - TSE (6 securities) ====================
    # High intraday liquidity + high price fluctuation
    _sec_jp('9984'),      # SoftBank Group   - tech/investment, beta ~1.5, avg intraday range 2-3%
    _sec_jp('8035'),      # Tokyo Electron   - semiconductor equipment, very volatile, range 2-4%
    _sec_jp('6857'),      # Advantest        - semiconductor test, high volatility, range 2-4%
    _sec_jp('6920'),      # Lasertec         - semiconductor inspection, extreme volatility, range 3-5%
    _sec_jp('9983'),      # Fast Retailing   - Uniqlo, heavy Nikkei weight, range 1.5-3%
    _sec_jp('6758'),      # Sony Group       - diversified tech/entertainment, liquid, range 1.5-2.5%
    # ==================== HONG KONG - HKEX (6 securities) ================
    # High intraday liquidity + high price fluctuation
    _sec_hk('9988'),      # Alibaba Group    - e-commerce/cloud, very volatile, range 2-4%
    _sec_hk('700'),       # Tencent Holdings - tech/gaming, most liquid on HKEX, range 1.5-3%
    _sec_hk('3690'),      # Meituan          - delivery/tech, volatile, range 2-4%
    _sec_hk('9618'),      # JD.com           - e-commerce, volatile, range 2-4%
    _sec_hk('1810'),      # Xiaomi           - electronics/EV, high retail volume, range 2-3%
    _sec_hk('1211'),      # BYD              - EV/batteries, volatile, range 2-3%
]

logLevel = logging.DEBUG
#logLevel = logging.INFO
MODE = 'INIT_DB' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
periods = ['1Min'] #periods = ['1Min','30Min']
numDaysHistCandles = 89

calibrationPauseSeconds = 900  # 15 min
calibration_timezone = 'America/New_York'
calibration_active_hours = (0, 8)

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
exitTimeSeconds = 11400  # 190 * 60
stopLossCoefficient = 20
correction = 0.0
spread = 0.0

openaikey = platform['secrets']['openaikey']
