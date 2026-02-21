#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime as dt
import pytz
import logging
from Configuration import TradingPlatfomSettings as tps

platform = tps.platform
securities = []
securities = [
    {
        'seccode': 'AAPL'
        ,'board': 'EQTY'
        ,'market': 'NASDAQ'
        ,'decimals' : 2 
        ,'id' : 0
        ,'params': {
            'algorithm': 'RsiAndAtr',
            'entryByMarket': False,
            'exitTimeSeconds': 11400,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51
            ,"positionMargin": 0.002 
            ,"stopLossCoefficient": 2 
            ,"acceptableTrainingError": 0.000192
            ,'period': '1Min'
            ,'rsiCoeff': '9'
        }
    }
    ,{
        'seccode': 'INTC'       
        ,'board': 'EQTY'
        ,'market': 'NASDAQ'
        ,'decimals' : 2
        ,'id' : 0
        ,'params': {
            'algorithm': 'RsiAndAtr',
            'entryByMarket': False,
            'exitTimeSeconds': 11400,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51
            ,"longPositionMargin": 0.002 
            ,"shortPositionMargin": 0.002 
            ,"stopLossCoefficient": 2 
            ,"acceptableTrainingError": 0.000192
            ,'period': '1Min'
            ,'rsiCoeff': '9'
        }
    }
    ,{
        'seccode': 'NVDA'        
        ,'board': 'EQTY'
        ,'market': 'NASDAQ'
        ,'decimals' : 2
        ,'id' : 0
        ,'params': {
            'algorithm': 'RsiAndAtr',
            'entryByMarket': False,
            'exitTimeSeconds': 11400,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51
            ,"longPositionMargin": 0.002 
            ,"shortPositionMargin": 0.002 
            ,"stopLossCoefficient": 2 
            ,"acceptableTrainingError": 0.000192
            ,'period': '1Min'
            ,'rsiCoeff': '9'
        }
    }
]
logLevel = logging.DEBUG
#logLevel = logging.INFO
MODE = 'TEST_OFFLINE' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
periods = ['1Min'] #periods = ['1Min','30Min']
numDaysHistCandles = 89

calibrationPauseSeconds = 900  # 15 min

simulation_net_balance = 29000

current_tz = pytz.timezone('America/New_York')
# Localize the 'since' and 'until' datetime objects to the specified timezone
since = current_tz.localize(dt.datetime.now() - dt.timedelta(days=numDaysHistCandles))
until = current_tz.localize(dt.datetime.now())
#until = current_tz.localize(dt.datetime.now())
between_time = (
    current_tz.localize(dt.datetime.strptime('07:00', '%H:%M')).time(),
    current_tz.localize(dt.datetime.strptime('23:40', '%H:%M')).time()
)
tradingTimes = (dt.time(9, 45), dt.time(15, 45))

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
factorPosition_Balance = 0.3
factorMargin_Position  = 0.001
entryTimeSeconds = 60
exitTimeSeconds = 11400  # 190 * 60
stopLossCoefficient = 2
correction = 0.0
spread = 0.0
time2close = dt.time(15, 46)  # Definido como 16:30 (4:30 PM)

openaikey = platform['secrets']['openaikey']
