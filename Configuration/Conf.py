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
         time2close=dt.time(15, 53),
         company_name=None, sector=None, beta_info=None, volatility_range=None):
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
        'fallback_source': 'yfinance',
        'fallback_ticker': code,
        'company_name': company_name,
        'sector': sector,
        'beta_info': beta_info,
        'volatility_range': volatility_range,
        'params': dict(_BASE_PARAMS),
    }

_YAHOO_EU_SUFFIX = {
    'XETRA': '.DE', 'BME': '.MC', 'SBF': '.PA',
    'BVME': '.MI', 'LSE': '.L', 'AEB': '.AS',
}

def _sec_eu(code, decimals=2, market='XETRA', timezone='Europe/Berlin',
            currency='EUR', exchange='SMART', primary_exchange='IBIS',
            trading_times=(dt.time(9, 16), dt.time(15, 30)),
            time2close=dt.time(15, 38),
            company_name=None, sector=None, beta_info=None, volatility_range=None,
            **extra):
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
        'fallback_source': 'yfinance',
        'fallback_ticker': f"{code}{_YAHOO_EU_SUFFIX.get(market, '.DE')}",
        'company_name': company_name,
        'sector': sector,
        'beta_info': beta_info,
        'volatility_range': volatility_range,
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
            time2close=dt.time(15, 25),
            company_name=None, sector=None, beta_info=None, volatility_range=None):
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
        'fallback_source': 'yfinance',
        'fallback_ticker': f'{code}.T',
        'company_name': company_name,
        'sector': sector,
        'beta_info': beta_info,
        'volatility_range': volatility_range,
        'params': dict(_BASE_PARAMS),
    }

# ---------------------------------------------------------------------------
# Hong Kong (HKEX) helper
# Trading hours: 9:30-12:00 (morning) + 13:00-16:00 (afternoon), lunch break
# IB: exchange SMART, primaryExchange SEHK, currency HKD
# ---------------------------------------------------------------------------
def _sec_hk(code, decimals=2, board_lot=100,
            trading_times=(dt.time(9, 35), dt.time(15, 55)),
            time2close=dt.time(15, 58),
            company_name=None, sector=None, beta_info=None, volatility_range=None):
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
        'fallback_source': 'yfinance',
        'fallback_ticker': f'{int(code):04d}.HK',
        'company_name': company_name,
        'sector': sector,
        'beta_info': beta_info,
        'volatility_range': volatility_range,
        'params': dict(_BASE_PARAMS),
    }

securities = [
    # ==================== AMERICAS - NASDAQ ====================
    _sec('TSLA', company_name='Tesla', sector='EV/energy', beta_info='very volatile', volatility_range='2-4%'),
    _sec('AMD', company_name='AMD', sector='semiconductor', beta_info='beta ~1.7', volatility_range='2-3%'),
    _sec('AAPL', company_name='Apple', sector='tech megacap', beta_info='liquid', volatility_range='1-2%'),
    _sec('INTC', decimals=3, company_name='Intel', sector='semiconductor', beta_info='moderate volatility', volatility_range='1.5-2.5%'),
    _sec('NVDA', company_name='NVIDIA', sector='AI/GPU', beta_info='very volatile', volatility_range='2-4%'),
    _sec('SOFI', company_name='SoFi', sector='fintech', beta_info='beta ~1.8', volatility_range='3-5%'),
    _sec('MARA', company_name='MARA Holdings', sector='bitcoin mining', beta_info='extreme volatility', volatility_range='4-8%'),
    _sec('RIVN', company_name='Rivian', sector='EV startup', beta_info='volatile', volatility_range='3-5%'),
    _sec('HOOD', company_name='Robinhood', sector='fintech/trading', beta_info='volatile', volatility_range='3-5%'),
    _sec('SMCI', company_name='Super Micro', sector='AI servers', beta_info='extreme volatility', volatility_range='4-8%'),
    _sec('DKNG', company_name='DraftKings', sector='sports betting', beta_info='volatile', volatility_range='2-4%'),
    _sec('MSTR', company_name='MicroStrategy', sector='bitcoin treasury', beta_info='extreme volatility', volatility_range='4-8%'),
    _sec('AMZN', company_name='Amazon', sector='e-commerce/cloud', beta_info='liquid', volatility_range='1.5-2.5%'),
    _sec('MSFT', company_name='Microsoft', sector='tech megacap', beta_info='liquid', volatility_range='1-2%'),
    _sec('PENN', company_name='Penn Entertainment', sector='sports betting', beta_info='beta ~2.0', volatility_range='3-5%'),
    _sec('AFRM', company_name='Affirm', sector='fintech/BNPL', beta_info='beta ~2.5', volatility_range='3-5%'),
    _sec('PLTR', company_name='Palantir', sector='AI/data analytics', beta_info='beta ~2.5', volatility_range='3-5%'),
    _sec('SHOP', primary_exchange='NYSE', company_name='Shopify', sector='e-commerce platform', beta_info='beta ~2.0', volatility_range='2-4%'),
    _sec('RBLX', company_name='Roblox', sector='gaming/metaverse', beta_info='beta ~2.0', volatility_range='2-4%'),
    _sec('CRWD', company_name='CrowdStrike', sector='cybersecurity', beta_info='beta ~1.5', volatility_range='2-3%'),
    _sec('SNAP', primary_exchange='NYSE', company_name='Snap', sector='social media/AR', beta_info='beta ~1.5', volatility_range='2-4%'),
    _sec('ROKU', company_name='Roku', sector='streaming tech', beta_info='beta ~2.0', volatility_range='3-5%'),
    _sec('ENPH', company_name='Enphase Energy', sector='solar/clean energy', beta_info='beta ~1.8', volatility_range='3-5%'),
    _sec('NET', primary_exchange='NYSE', company_name='Cloudflare', sector='cloud/cybersecurity', beta_info='beta ~1.5', volatility_range='2-4%'),
    _sec('MRNA', company_name='Moderna', sector='biotech/vaccines', beta_info='beta ~1.8', volatility_range='3-5%'),
    _sec('FSLR', company_name='First Solar', sector='solar manufacturing', beta_info='beta ~1.5', volatility_range='2-4%'),
    _sec('ALB', primary_exchange='NYSE', company_name='Albemarle', sector='lithium/chemicals', beta_info='beta ~1.5', volatility_range='2-4%'),
    _sec('DASH', company_name='DoorDash', sector='delivery platform', beta_info='beta ~1.3', volatility_range='2-3%'),
    # ==================== EUROPE ====================
    # Germany - XETRA
    _sec_eu('SBX', company_name='Stabilus', sector='industrial', beta_info='moderate volatility', volatility_range='1.5-2.5%'),
    _sec_eu('IFX', company_name='Infineon', sector='semiconductor', beta_info='beta 1.83', volatility_range='2-3%'),
    _sec_eu('DBK', company_name='Deutsche Bank', sector='banking', beta_info='beta 1.46', volatility_range='1.5-2.5%'),
    _sec_eu('ENR', company_name='Siemens Energy', sector='energy', beta_info='beta 1.60-1.81', volatility_range='2-3%'),
    # Spain - BME
    _sec_eu('BBVA', market='BME', timezone='Europe/Madrid', primary_exchange='BM',
            company_name='BBVA', sector='banking', beta_info='beta 1.25', volatility_range='1.5-2.5%'),
    _sec_eu('SAN', market='BME', timezone='Europe/Madrid', primary_exchange='BM',
            company_name='Santander', sector='banking', beta_info='beta 1.20', volatility_range='1.5-2.5%'),
    # France - Euronext Paris
    _sec_eu('GLE', market='SBF', timezone='Europe/Paris', primary_exchange='SBF',
            company_name='Societe Generale', sector='banking', beta_info='beta 1.39', volatility_range='2-3%'),
    # Italy - Borsa Italiana
    _sec_eu('UCG', market='BVME', timezone='Europe/Rome', primary_exchange='BVME',
            company_name='UniCredit', sector='banking', beta_info='beta 1.28', volatility_range='2-3%'),
    # UK - London Stock Exchange
    _sec_eu('BARC', market='LSE', timezone='Europe/London', currency='GBP',
            primary_exchange='LSE', fx_currency='GBX',
            trading_times=(dt.time(9, 46), dt.time(14, 40)),
            time2close=dt.time(14, 45),
            company_name='Barclays', sector='banking', beta_info='beta 1.98', volatility_range='2-3%'),
    # Netherlands - Euronext Amsterdam
    _sec_eu('ASML', market='AEB', timezone='Europe/Amsterdam', primary_exchange='AEB',
            company_name='ASML Holding', sector='semiconductor equip', beta_info='beta ~1.3', volatility_range='2-3%'),
    # France - Euronext Paris
    _sec_eu('BNP', market='SBF', timezone='Europe/Paris', primary_exchange='SBF',
            company_name='BNP Paribas', sector='banking', beta_info='beta ~1.3', volatility_range='1.5-2.5%'),
    # Germany - XETRA
    _sec_eu('CBK', company_name='Commerzbank', sector='banking', beta_info='beta ~1.4', volatility_range='2-3%'),
    # UK - London Stock Exchange
    _sec_eu('FLTR', market='LSE', timezone='Europe/London', currency='GBP',
            primary_exchange='LSE', fx_currency='GBX',
            trading_times=(dt.time(9, 46), dt.time(14, 40)),
            time2close=dt.time(14, 45),
            company_name='Flutter Entertainment', sector='sports betting', beta_info='beta ~1.3', volatility_range='2-3%'),
    _sec_eu('AIR', market='SBF', timezone='Europe/Paris', primary_exchange='SBF',
            company_name='Airbus', sector='aerospace/defense', beta_info='beta ~1.3', volatility_range='1.5-2.5%'),
    _sec_eu('ADS', company_name='Adidas', sector='sportswear', beta_info='beta ~1.3', volatility_range='2-3%'),
    _sec_eu('RNO', market='SBF', timezone='Europe/Paris', primary_exchange='SBF',
            company_name='Renault', sector='automotive', beta_info='beta ~1.5', volatility_range='2-3%'),
    _sec_eu('TKA', company_name='ThyssenKrupp', sector='industrial/steel', beta_info='beta ~1.6', volatility_range='2-4%'),
    _sec_eu('VOW3', company_name='Volkswagen', sector='automotive', beta_info='beta ~1.3', volatility_range='2-3%'),
    _sec_eu('SAP', company_name='SAP SE', sector='enterprise software', beta_info='beta ~1.1', volatility_range='1.5-2.5%'),
    _sec_eu('BAS', company_name='BASF', sector='chemicals', beta_info='beta ~1.2', volatility_range='1.5-2.5%'),
    _sec_eu('BMW', company_name='BMW', sector='automotive', beta_info='beta ~1.3', volatility_range='1.5-2.5%'),
    _sec_eu('TTE', market='SBF', timezone='Europe/Paris', primary_exchange='SBF',
            company_name='TotalEnergies', sector='energy', beta_info='beta ~1.2', volatility_range='1.5-2.5%'),
    _sec_eu('DHL', company_name='DHL Group', sector='logistics', beta_info='beta ~1.2', volatility_range='1.5-2.5%'),
    # ==================== JAPAN - TSE ====================
    _sec_jp('7201', company_name='Nissan Motor', sector='automotive', volatility_range='2.5%'),
    _sec_jp('3350', company_name='Metaplanet', sector='Bitcoin proxy', volatility_range='8.1%'),
    _sec_jp('9432', company_name='NTT', sector='telecom', volatility_range='2.0%'),
    _sec_jp('4755', company_name='Rakuten Group', sector='e-commerce/fintech', volatility_range='2.0%'),
    _sec_jp('5401', company_name='Nippon Steel', sector='steel', volatility_range='2.0%'),
    _sec_jp('9434', company_name='SoftBank Corp', sector='telecom', volatility_range='1.5%'),
    _sec_jp('7261', company_name='Mazda Motor', sector='automotive', volatility_range='3.0%'),
    _sec_jp('8035', company_name='Tokyo Electron', sector='semiconductor equip', beta_info='very volatile', volatility_range='2-4%'),
    _sec_jp('6857', company_name='Advantest', sector='semiconductor test', beta_info='high volatility', volatility_range='2-4%'),
    _sec_jp('6920', company_name='Lasertec', sector='semiconductor inspect', beta_info='extreme volatility', volatility_range='3-5%'),
    _sec_jp('9983', company_name='Fast Retailing', sector='retail/Uniqlo', beta_info='heavy Nikkei weight', volatility_range='1.5-3%'),
    _sec_jp('6758', company_name='Sony Group', sector='tech/entertainment', beta_info='liquid', volatility_range='1.5-2.5%'),
    _sec_jp('7974', company_name='Nintendo', sector='gaming/entertainment', beta_info='liquid', volatility_range='2-3%'),
    _sec_jp('6861', company_name='Keyence', sector='sensors/automation', beta_info='beta ~1.3', volatility_range='1.5-2.5%'),
    _sec_jp('7267', company_name='Honda Motor', sector='automotive', beta_info='liquid', volatility_range='1.5-2.5%'),
    _sec_jp('7203', company_name='Toyota Motor', sector='automotive', beta_info='very liquid', volatility_range='1.5-2.5%'),
    _sec_jp('4568', company_name='Daiichi Sankyo', sector='pharma', beta_info='volatile', volatility_range='2-4%'),
    # ==================== HONG KONG - HKEX ====================
    _sec_hk('9866', company_name='NIO Inc', sector='EV', volatility_range='4-5%'),
    _sec_hk('1024', company_name='Kuaishou', sector='social/video', volatility_range='3%'),
    _sec_hk('2015', company_name='Li Auto', sector='EV', volatility_range='3-4%'),
    _sec_hk('9988', company_name='Alibaba Group', sector='e-commerce/cloud', beta_info='very volatile', volatility_range='2-4%'),
    _sec_hk('700', company_name='Tencent', sector='tech/gaming', beta_info='most liquid HKEX', volatility_range='1.5-3%'),
    _sec_hk('3690', company_name='Meituan', sector='delivery/tech', beta_info='volatile', volatility_range='2-4%'),
    _sec_hk('9618', company_name='JD.com', sector='e-commerce', beta_info='volatile', volatility_range='2-4%'),
    _sec_hk('1810', board_lot=200, company_name='Xiaomi', sector='electronics/EV', beta_info='high retail volume', volatility_range='2-3%'),
    _sec_hk('1211', board_lot=500, company_name='BYD', sector='EV/batteries', beta_info='volatile', volatility_range='2-3%'),
    _sec_hk('9888', company_name='Baidu', sector='AI/search', beta_info='volatile', volatility_range='2-4%'),
    _sec_hk('9999', company_name='NetEase', sector='gaming/entertainment', volatility_range='2-3%'),
    _sec_hk('9626', company_name='Bilibili', sector='video/gaming platform', beta_info='beta ~1.5', volatility_range='3-5%'),
    _sec_hk('175', board_lot=1000, company_name='Geely Auto', sector='automotive', beta_info='volatile', volatility_range='2-4%'),
    _sec_hk('2269', board_lot=500, company_name='WuXi Biologics', sector='biotech/CDMO', beta_info='volatile', volatility_range='3-5%'),
]

#logLevel = logging.DEBUG
logLevel = logging.INFO
MODE = 'OPERATIONAL' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
periods = ['1Min'] #periods = ['1Min','30Min']
numDaysHistCandles = 3
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
# Conservative fill simulation: require directional cross + slippage
CALIBRATION_FILL_SLIPPAGE = 0.0001      # LMT pullback fill slippage
CALIBRATION_BREAKOUT_SLIPPAGE = 0.0005  # Market-like breakout fill slippage (5x LMT)
CALIBRATION_MAX_VOLUME_PARTICIPATION = 0.10
# Walk-forward: train on first 67%, validate on last 33% (prevents overfitting)
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
