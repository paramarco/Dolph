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
    'period': '1Min',
    # ===== 5 META PARAMS (optimizable, derive old indicator/margin/SL params) =====
    'EMA_BASE': 14,         # derives EMA_FAST, EMA_MID, EMA_SLOW
    'VOL_WINDOW': 20,       # derives ATR_PERIOD, BB_WINDOW, BB_PERCENTILE_WINDOW, ATR_SLOPE_WINDOW, FVP_WINDOW
    'TREND_WINDOW': 14,     # derives ADX_PERIOD
    'TP_MULT': 1.5,         # margin = TP_MULT * max(ATR/close, BB_width)
    'SL_RR': 2.0,           # derives stopLossCoefficient
    # ===== 5 DIRECTLY OPTIMIZABLE PARAMS =====
    'VCP_ATR_SLOPE_EXPANSION': 0.03,
    'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.245,
    'VCP_ADX_TREND_THRESHOLD': 18,
    'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
    'MIN_RELATIVE_VOLUME': 0.8,
    # ===== DERIVED PARAMS (set by _derive_params(), do not optimize) =====
    'EMA_FAST': 14,         # = EMA_BASE
    'EMA_MID': 21,          # = round(1.5 * EMA_BASE)
    'EMA_SLOW': 35,         # = round(2.5 * EMA_BASE)
    'ATR_PERIOD': 20,       # = VOL_WINDOW
    'BB_WINDOW': 20,        # = VOL_WINDOW
    'BB_STD': 2,
    'BB_PERCENTILE_WINDOW': 60,  # = 3 * VOL_WINDOW
    'ATR_SLOPE_WINDOW': 6,  # = max(3, VOL_WINDOW // 3)
    'FVP_WINDOW': 40,       # = 2 * VOL_WINDOW
    'ADX_PERIOD': 14,       # = TREND_WINDOW
    'stopLossCoefficient': 2.0,          # = SL_RR
    # ===== FROZEN RSI (valores restrictivos) =====
    'RSI_PERIOD': 14,
    'EXPANSION_RSI_SHORT_MIN': 45,
    'EXPANSION_RSI_LONG_MAX': 60,
    'TREND_RSI_LONG_MIN': 35,
    'TREND_RSI_LONG_MAX': 70,
    'TREND_RSI_SHORT_MIN': 30,
    'TREND_RSI_SHORT_MAX': 65,
    # (margin clamps removed — unified formula: TP_MULT * max(ATR/close, BB_width))
    # ===== FROZEN CALIBRATION PARAMS =====
    'CALIBRATION_LOOKBACK_DAYS': 90,
    'CALIBRATION_LIMIT_RESULTS': 40000,
    'CALIBRATION_MIN_ROWS': 1000,
    'CALIBRATION_MARGIN_MIN': 0.001,
    'CALIBRATION_MARGIN_MAX': 0.015,
    'CALIBRATION_MARGIN_STEPS': 10,
    'CALIBRATION_LOOKAHEAD_BARS': 60,
    # ===== FROZEN VOLUME ANALYSIS =====
    'VOLUME_AVG_WINDOW': 13,
    'VOLUME_SLOPE_WINDOW': 4,
    'BIG_VOLUME_THRESHOLD': 1.5,
    'EXTREME_VOLUME_THRESHOLD': 2.5,
    'BIG_BODY_ATR_THRESHOLD': 0.8,
    'EXTREME_BODY_ATR_THRESHOLD': 3.157,
    'DIVERGENCE_LOOKBACK': 10,
    # ===== FROZEN BUYING CLIMAX (effectively disabled) =====
    'BUYING_CLIMAX_LOOKBACK': 10,
    'BUYING_CLIMAX_TREND_LOOKBACK': 7,
    'BUYING_CLIMAX_EXTENSION': 1.5,
    'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
    # ===== FROZEN POSITION / CONFIDENCE / MOMENTUM =====
    'POSITION_COOLDOWN_SECONDS': 300,
    # (NO_VOLUME_CONFIDENCE_PENALTY removed — absorbed by mandatory/optional architecture)
    'MOMENTUM_LOOKBACK': 5,
    'COUNTER_TREND_THRESHOLD': 0.003,
    'COUNTER_TREND_FACTOR': 10.0,
    'SIGNAL_STABILITY_REQUIRED': 2,
    # ===== FROZEN LIQUIDITY / SMC =====
    'LIQ_BOS_LOOKBACK': 20,
    'LIQ_SWEEP_LOOKBACK': 5,
    'LIQ_ZONE_TOLERANCE': 0.005,
    'LIQ_SCORE_BONUS': 2,
    'LIQ_ADX_MIN': 15,
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
        'board_lot': 1,
        'params': dict(_BASE_PARAMS),
    }


def _sec_eu(code, decimals=2, market='XETRA', timezone='Europe/Berlin',
            currency='EUR', exchange='SMART', primary_exchange='IBIS',
            trading_times=(dt.time(9, 16), dt.time(15, 30)),
            time2close=dt.time(15, 38), **extra):
    d = {
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
        'board_lot': 1,
        'params': dict(_BASE_PARAMS),
    }
    d.update(extra)
    return d

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
        'board_lot': 100,
        'params': dict(_BASE_PARAMS),
    }

# ---------------------------------------------------------------------------
# Hong Kong (HKEX) helper
# Trading hours: 9:30-12:00 (morning) + 13:00-16:00 (afternoon), lunch break
# IB: exchange SMART, primaryExchange SEHK, currency HKD
# ---------------------------------------------------------------------------
def _sec_hk(code, decimals=2, board_lot=100,
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
        'board_lot': board_lot,
        'params': dict(_BASE_PARAMS),
    }

securities = [
    # ==================== AMERICAS - NASDAQ (14 securities) ====================
    # _sec('TSLA'),               # Tesla          - EV/energy, very volatile, range 2-4%
    # _sec('AMD'),                # AMD            - semiconductor, high beta ~1.7, range 2-3%
    # _sec('AAPL'),               # Apple          - tech megacap, liquid, range 1-2%
    # _sec('INTC', decimals=3),   # Intel          - semiconductor, moderate volatility, range 1.5-2.5%
    # _sec('NVDA'),               # NVIDIA         - AI/GPU, very volatile, range 2-4%
    # _sec('SOFI'),               # SoFi           - fintech, high beta ~1.8, range 3-5%
    # _sec('MARA'),               # MARA Holdings  - bitcoin mining, extreme volatility, range 4-8%
    # _sec('RIVN'),               # Rivian         - EV startup, volatile, range 3-5%
    # _sec('HOOD'),               # Robinhood      - fintech/trading, volatile, range 3-5%
    # _sec('SMCI'),               # Super Micro    - AI servers, extreme volatility, range 4-8%
    # _sec('DKNG'),               # DraftKings     - sports betting, volatile, range 2-4%
    # _sec('MSTR'),               # MicroStrategy  - bitcoin treasury, extreme volatility, range 4-8%
    # _sec('AMZN'),               # Amazon         - e-commerce/cloud, liquid, range 1.5-2.5%
    # _sec('MSFT'),               # Microsoft      - tech megacap, liquid, range 1-2%
    # _sec('RBLX'),               # Roblox         - gaming/metaverse, beta ~2.0, range 2-4%
    # _sec('CRWD'),               # CrowdStrike    - cybersecurity, beta ~1.5, range 2-3%
    # _sec('SNAP', primary_exchange='NYSE'),  # Snap           - social media/AR, beta ~1.5, range 2-4%
    # _sec('ROKU'),               # Roku           - streaming tech, beta ~2.0, range 3-5%
    # _sec('ENPH'),               # Enphase Energy - solar/clean energy, beta ~1.8, range 3-5%
    # _sec('NET', primary_exchange='NYSE'),   # Cloudflare     - cloud/cybersecurity, beta ~1.5, range 2-4%
    # _sec('MRNA'),               # Moderna        - biotech/vaccines, beta ~1.8, range 3-5%
    # _sec('FSLR'),               # First Solar    - solar manufacturing, beta ~1.5, range 2-4%
    # _sec('ALB', primary_exchange='NYSE'),   # Albemarle      - lithium/chemicals, beta ~1.5, range 2-4%
    # _sec('DASH'),               # DoorDash       - delivery platform, beta ~1.3, range 2-3%
    # ==================== EUROPE (12 securities) ====================
    # # Germany - XETRA
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
    # _sec_eu('AIR', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),    # Airbus         - aerospace/defense, beta ~1.3, range 1.5-2.5%
    # _sec_eu('ADS'),             # Adidas         - sportswear, beta ~1.3, range 2-3%
    # _sec_eu('RNO', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),    # Renault        - automotive, beta ~1.5, range 2-3%
    # _sec_eu('TKA'),             # ThyssenKrupp   - industrial/steel, beta ~1.6, range 2-4%
    # _sec_eu('VOW3'),            # Volkswagen Pref- automotive, beta ~1.3, range 2-3%
    # _sec_eu('SAP'),             # SAP SE         - enterprise software, beta ~1.1, range 1.5-2.5%
    # _sec_eu('BAS'),             # BASF           - chemicals, beta ~1.2, range 1.5-2.5%
    # _sec_eu('BMW'),             # BMW            - automotive, beta ~1.3, range 1.5-2.5%
    # _sec_eu('TTE', market='SBF', timezone='Europe/Paris', primary_exchange='SBF'),    # TotalEnergies  - energy, beta ~1.2, range 1.5-2.5%
    # _sec_eu('DHL'),             # DHL Group      - logistics, beta ~1.2, range 1.5-2.5%
    # ==================== JAPAN - TSE (affordable, lot_cost ≤ $800 USD) ====================
    _sec_jp('7201'),              # Nissan Motor   - automotive, ¥380, lot=$255, vol 2.5%
    _sec_jp('3350'),              # Metaplanet     - Bitcoin proxy, ¥373, lot=$250, vol 8.1%
    _sec_jp('9432'),              # NTT            - telecom, ¥155, lot=$104, vol 2.0%
    _sec_jp('4755'),              # Rakuten Group  - e-commerce/fintech, ¥800, lot=$537, vol 2.0%
    _sec_jp('5401'),              # Nippon Steel   - steel, ¥601, lot=$403, vol 2.0%
    _sec_jp('9434'),              # SoftBank Corp  - telecom, ¥213, lot=$143, vol 1.5%
    _sec_jp('7261'),              # Mazda Motor    - automotive, ¥1200, lot=$805, vol 3.0%
    # --- Too expensive for $900/trade (commented) ---
    # _sec_jp('9984'),             # SoftBank Group  - ¥9000, lot=$6040, TOO EXPENSIVE
    # _sec_jp('8035'),             # Tokyo Electron  - ¥22000, lot=$14765, TOO EXPENSIVE
    # _sec_jp('6857'),             # Advantest       - ¥7500, lot=$5034, TOO EXPENSIVE
    # _sec_jp('6920'),             # Lasertec        - ¥12000, lot=$8054, TOO EXPENSIVE
    # _sec_jp('9983'),             # Fast Retailing  - ¥50000, lot=$33557, TOO EXPENSIVE
    # _sec_jp('6758'),             # Sony Group      - ¥3200, lot=$2148, TOO EXPENSIVE
    # _sec_jp('7974'),             # Nintendo        - ¥10000, lot=$6711, TOO EXPENSIVE
    # _sec_jp('6861'),             # Keyence         - ¥65000, lot=$43624, TOO EXPENSIVE
    # _sec_jp('7267'),             # Honda Motor     - ¥1400, lot=$940, TOO EXPENSIVE
    # _sec_jp('7203'),             # Toyota Motor    - ¥2800, lot=$1879, TOO EXPENSIVE
    # _sec_jp('4568'),             # Daiichi Sankyo  - ¥2500, lot=$1678, TOO EXPENSIVE
    # ==================== HONG KONG - HKEX (affordable, lot_cost ≤ $800 USD) ====================
    _sec_hk('9866'),              # NIO Inc        - EV, HK$30, lot=100, lot_cost=$384, vol 4-5%
    _sec_hk('1024'),              # Kuaishou       - social/video, HK$45, lot=100, lot_cost=$576, vol 3%
    _sec_hk('2015'),              # Li Auto        - EV, HK$63, lot=100, lot_cost=$807, vol 3-4%
    # --- Too expensive for $900/trade (commented) ---
    # _sec_hk('9988'),             # Alibaba       - HK$125, lot=100, lot_cost=$1601, TOO EXPENSIVE
    # _sec_hk('700'),              # Tencent        - HK$430, lot=100, lot_cost=$5506, TOO EXPENSIVE
    # _sec_hk('3690'),             # Meituan        - HK$140, lot=100, lot_cost=$1793, TOO EXPENSIVE
    # _sec_hk('9618'),             # JD.com         - HK$130, lot=100, lot_cost=$1665, TOO EXPENSIVE
    # _sec_hk('1810', board_lot=200),  # Xiaomi     - HK$47, lot=200, lot_cost=$1203, TOO EXPENSIVE
    # _sec_hk('1211', board_lot=500),  # BYD        - HK$350, lot=500, lot_cost=$22407, TOO EXPENSIVE
    # _sec_hk('9888'),             # Baidu          - HK$80, lot=100, lot_cost=$1024, TOO EXPENSIVE
    # _sec_hk('9999'),             # NetEase        - HK$150, lot=100, lot_cost=$1921, TOO EXPENSIVE
    # _sec_hk('9626'),             # Bilibili       - HK$120, lot=100, lot_cost=$1537, TOO EXPENSIVE
    # _sec_hk('175', board_lot=1000),  # Geely Auto - HK$16, lot=1000, lot_cost=$2049, TOO EXPENSIVE
    # _sec_hk('2269', board_lot=500),  # WuXi Bio   - HK$15, lot=500, lot_cost=$961, TOO EXPENSIVE
]

logLevel = logging.DEBUG
#logLevel = logging.INFO
MODE = 'TEST_OFFLINE' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
periods = ['1Min'] #periods = ['1Min','30Min']
numDaysHistCandles = 89
calibrationPauseSeconds = 3600  # 1 hour
simulation_net_balance = 5000
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
# Multi-resolution coordinate descent step sizes (coarse → fine)
CALIBRATION_STEP_SIZES = [0.30, 0.15, 0.08]
# Stochastic perturbation to escape local optima
CALIBRATION_PERTURB_RANGE = 0.15    # Random ±15% perturbation amplitude
CALIBRATION_MAX_PERTURBS = 3        # Max failed perturbation attempts before giving up
MIN_CALIBRATION_SCORE = 100.0
# Idea #3: Volume confirmation gate (module-level fallback)
MIN_RELATIVE_VOLUME = 0.8
# Idea #4: Margin dynamic cost floor
MIN_ABS_MARGIN_MULTIPLIER = 1.5
# (EXPANSION_SCORE_MULT, TREND_SCORE_MULT removed — mandatory/optional replaces score thresholds)
# Idea #6: Trailing TP in simulation
TRAILING_TP_ENABLED = True
TRAILING_TP_RETRACE = 0.50
# SL aversion: proportional penalty on real SL losses during calibration
CALIBRATION_SL_AVERSION = 0.15
CALIBRATION_DD_AVERSION = 0.10
CALIBRATION_MIN_TRADES_PER_DAY = 0.5
CALIBRATION_FILL_SLIPPAGE = 0.0001      # LMT pullback fill slippage
CALIBRATION_BREAKOUT_SLIPPAGE = 0.0005  # Market-like breakout fill slippage (5x LMT)
CALIBRATION_MAX_VOLUME_PARTICIPATION = 0.10
CALIBRATION_TRAIN_RATIO = 0.67
CALIBRATION_TEST_WEIGHT = 0.40

# Approximate USD exchange rates for calibration simulation position sizing
FX_RATES_FROM_USD = {
    'USD': 1.0,
    'EUR': 0.92,
    'GBP': 0.79,
    'GBX': 79.0,   # GBP pence: LSE stocks priced in pence (0.79 * 100)
    'JPY': 149.0,
    'HKD': 7.81,
}
