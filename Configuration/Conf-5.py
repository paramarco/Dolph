#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime as dt
import pytz
import logging
from Configuration import TradingPlatfomSettings as tps

platform = tps.platform

SECURITY_TZ_FILTER = None  # OPERATIONAL: load all securities from DB

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
factorPosition_Balance = 0.33    # DolphRobot.py:436, MinerviniClaude.py:956 (direct cm.X usage)

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
MIN_CONFIDENCE_FILTER = 0.70
# Idea #3: Volume confirmation gate (module-level fallback)
MIN_RELATIVE_VOLUME = 0.8
# Idea #4: Margin dynamic cost floor
MIN_ABS_MARGIN_MULTIPLIER = 1.5
MARGIN_DAILY_RANGE_CAP = 0.28     # margin cap as fraction of avg daily range
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
# Wave frequency analysis for per-security GAUSS_MU
BUTTER_ORDER = 2         # Butterworth filter order
BUTTER_CUTOFF = 0.05     # low cutoff for smoother curve
PEAK_DISTANCE = 30       # minimum bars between peaks (30 min)
SWING_THRESHOLD = 0.95   # 95% of swing amplitude
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
